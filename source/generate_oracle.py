import json
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY"
)

def textualize_json(data):
    """Convert JSON data to readable text format"""
    if isinstance(data, list):
        return json.dumps(data, indent=2, ensure_ascii=False)
    elif isinstance(data, dict):
        return json.dumps(data, indent=2, ensure_ascii=False)
    else:
        return str(data)


def create_test_oracle_prompt(user_query, user_situation, persona_items, user_story_items,
                              domain_knowledge_items, background_items, nfr_items):
    """Create prompt for test oracle generation"""

    prompt = f"""You are analyzing a software system to infer its expected behavior.

Below is the structured context about the software:

- **User Input (current query):**
  {user_query}

- **User Situation:**
  {user_situation}

- **User Persona:**
  {textualize_json(persona_items)}

- **Relevant User Story:**
  {textualize_json(user_story_items)}

- **Domain Knowledge:**
  {textualize_json(domain_knowledge_items)}

- **Background Information:**
  {textualize_json(background_items)}

- **Non-Functional Requirements:**
  {textualize_json(nfr_items)}        

Using all the above information, describe **the expected behavior of the software** in our given format of test oracle — 
that is, how the system should respond or act according to the user's input and the provided context.

The output should follow the json format as:
{{
  "Title": "Short descriptive title of what this test verifies (string).",

  "TestObjective": "Concise description of the main goal or purpose of the test (string). It should define the functionality, feature, or system behavior being validated.",

  "Preconditions": [
    "List of environment or system states required before running the test (array of strings).",
    "May include configurations, datasets, services, user accounts, or hardware setup."
  ],

  "TestInputs": {{
    "Description": "Narrative description of what input data or actions are provided to the system (string).",
    "Source": "Origin of the inputs, such as file path, API endpoint, simulated user entry (string).",
    "Format": "Format or data type of the input (string). Examples: JSON, CSV, Image, Text.",
    "InputData": {{
      "Key": "Placeholder for input name or identifier (string).",
      "Value": "Placeholder for input content, value, or parameter (string, number, or object)."
    }},
    "ExpectedInputConditions": [
      "Rules or constraints that must be met by the input data (array of strings).",
      "Used to validate that the test input itself is valid before execution."
    ]
  }},

  "ExpectedResults": {{
    "Description": "Narrative explanation of what the correct output or behavior should be (string).",
    "Oracle": {{
      "Purpose": "Formal or structured definition of the expected result (object). This is the 'test oracle' specifying correct system behavior.",
      "Structure": "Defines expected output fields, attributes, or states. Can reference an external schema or include inline placeholders."
    }},
    "AcceptanceRules": [
      "Set of logical conditions or rules describing how expected results are derived (array of strings).",
      "Examples: thresholds, pattern matches, relationships, constraints, or computed properties."
    ]
  }},

  "PassFailCriteria": [
    "Explicit criteria that determine whether the test passes or fails (array of strings).",
    "Each rule compares actual system output with expected values or conditions.",
    "May include qualitative (e.g., keyword presence) or quantitative (e.g., accuracy ≥ 0.9) conditions."
  ],

  "Postconditions": [
    "System state or artifacts expected after test execution (array of strings).",
    "Describes persistent changes such as saved records, logs, notifications, or system resets."
  ],

  "Traceability": {{
    "RelatedRequirements": [
      "Requirement or design document IDs linked to this test case (array of strings)."
    ],
    "LinkedPersonas": [
      "User personas or roles relevant to the test context (array of strings)."
    ],
    "LinkedDocuments": [
      "Supporting domain or background documents referenced by this test (array of strings)."
    ]
  }},

  "Notes": "Free-text section for clarifications, assumptions, limitations, or contextual remarks relevant to this test case (string)."
}}

Your response (JSON only):"""

    return prompt


def generate_oracles_non_batch(enhanced_results_file, output_json, model="gpt-5-mini"):

    print("Loading enhanced results...")
    with open(enhanced_results_file, 'r') as f:
        enhanced_data = json.load(f)

    print(f"✓ Loaded {len(enhanced_data)} data points")

    results = []

    for data_idx, data_point in enumerate(enhanced_data):

        prompt = create_test_oracle_prompt(
            data_point.get('user_query', ''),
            data_point.get('user_situation', ''),
            data_point.get('persona', []),
            data_point.get('user_story', []),
            data_point.get('domain_knowledge', []),
            data_point.get('background', []),
            data_point.get('nfr', [])
        )

        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": "You are a software testing expert that generates detailed test oracles based on requirements and context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_output_tokens=10240
            )

            result_json = json.loads(response.output_text)

        except Exception as e:
            result_json = {"error": str(e)}

        results.append({
            "custom_id": f"oracle_{data_idx}",
            "response": result_json
        })

        # progressive saving
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Completed oracle_{data_idx}")

    return len(results)


if __name__ == '__main__':

    ENHANCED_RESULTS = "PATH_TO_YOUR_RETRIEVED_RESULTS"
    OUTPUT_RESULTS = "PATH_TO_YOUR_OUTPUT_DIR"
    MODEL = "gpt-5-mini"

    print("=" * 60)
    print("Test Oracle Generation - Non Batch Mode")
    print("=" * 60)

    generate_oracles_non_batch(ENHANCED_RESULTS, OUTPUT_RESULTS, MODEL)
