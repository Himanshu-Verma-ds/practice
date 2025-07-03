from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are a professional content writer. Your task is to create a high-quality article based on the following parameters:

- **Subject:** {subject}
- **Target Audience:** {audience}
- **Tone of Writing:** {tone}

Please structure the article with the following:
1. A strong and engaging introduction that hooks the reader.
2. At least two well-developed body paragraphs that explain the topic clearly and provide real-world examples or analogies.
3. A concise conclusion that summarizes key points and encourages further thinking or action.

Ensure the language is clear and accessible to the specified audience. Do not repeat content. Keep it informative and engaging.

Begin the article below:
------------------------
""",
    input_variables=["subject", "audience", "tone"],
    validate_template= True
)

prompt.save('prompt_template.json')