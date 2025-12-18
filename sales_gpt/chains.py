from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import BaseLLM
from sales_gpt.templates import STAGE_ANALYZER_INCEPTION_PROMPT_TEMPLATE, CONSULTANT_INCEPTION_PROMPT

class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        prompt = PromptTemplate(
            template=STAGE_ANALYZER_INCEPTION_PROMPT_TEMPLATE,
            input_variables=["conversation_history"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class ConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        prompt = PromptTemplate(
            template=CONSULTANT_INCEPTION_PROMPT,
            input_variables=[
                "consultant_name",
                "consultant_role",
                "company_name",
                "company_business",
                "company_values",
                "conversation_purpose",
                "conversation_stage",
                "conversation_history",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
