from typing import Any, Callable, Dict, List, Union
from pydantic import Field

from langchain.chains.base import Chain
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.agents.agent import AgentOutputParser
from langchain.prompts.base import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.llms import BaseLLM

from sales_gpt.chains import StageAnalyzerChain, ConversationChain
from sales_gpt.templates import SALES_AGENT_TOOLS_PROMPT
from sales_gpt.tools import get_tools

class CustomPromptTemplateForTools(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

class ConvoOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"  # change for salesperson_name
    verbose: bool = False

    def get_format_instructions(self) -> str:
        return "FORMAT INSTRUCTIONS" # Simplified, usually imported

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # Extract the final AI response, ignoring thoughts and actions
        response = text.split(f"{self.ai_prefix}:")[-1].strip()
        # Clean up <END_OF_TURN> if present in the middle (unlikely but safe)
        if "<END_OF_TURN>" in response:
            response = response.split("<END_OF_TURN>")[0]
            
        return AgentFinish({"output": response}, text)

    @property
    def _type(self) -> str:
        return "QA-agent"

class SalesGPT(Chain):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    current_conversation_stage: str = "1"
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    conversation_utterance_chain: ConversationChain = Field(...)

    Sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    use_tools: bool = True

    conversation_stage_dict: Dict = {
        "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone professional.",
        "2": "Qualification: Confirm if the prospect is the right person to discuss the fitness products we offer. Ensure they are the appropriate contact for this conversation.",
        "3": "Value Proposition: Highlight the best fitness products and accessories, emphasizing their unique features and benefits.",
        "4": "Needs Analysis: Ask open-ended questions to understand the prospect's fitness goals, challenges, and product preferences. Listen carefully and take detailed notes.",
        "5": "Solution Presentation: Customize our fitness product recommendations to address the prospect's specific needs and challenges identified during the conversation.",
        "6": "Objection Handling: Address any concerns or objections the prospect may have. Highlight the importance and effectiveness of our fitness products and solutions.",
        "7": "Close: Summarize the key points discussed and recommend the fitness product or accessory you believe is best suited for the user."
    }

    consultant_name: str = "AI Assistant"
    consultant_role: str = "Exercise Product Consultant"
    company_name: str = "Peloton"
    company_business: str = "Peloton is the leading provider of exercise accessories, or machines delivering unrivaled outputs, robust technology and expert services to fuel customers’ success through all phases of lifecycle."
    company_values: str = "We aim to empower our clients to achieve excellence in health by providing unparalleled machines and accessories, services throughout every stage of the lifecycle."
    conversation_purpose: str = "Collect user specifications pertaining to their desired exercise product, consolidate them, and suggest the best suited product."
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    def determine_conversation_stage(self):
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history='"\n"'.join(self.conversation_history),
            current_conversation_stage=self.current_conversation_stage,
        )

        self.current_conversation_stage = self.retrieve_conversation_stage(
            conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        # process human input
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    def step(self):
        self._call(inputs={})

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""

        # Generate agent's utterance
        if self.use_tools and self.Sales_agent_executor:
            try:
                ai_message = self.Sales_agent_executor.run(
                    input="",
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    consultant_name=self.consultant_name,
                    consultant_role=self.consultant_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                )
            except Exception as e:
                print(f"Agent Engine Error: {e}")
                # Fallback to simple chain if agent fails
                ai_message = self.conversation_utterance_chain.run(
                    consultant_name=self.consultant_name,
                    consultant_role=self.consultant_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_history="\n".join(self.conversation_history),
                    conversation_stage=self.current_conversation_stage,
                    conversation_type=self.conversation_type,
                )

        else:
            ai_message = self.conversation_utterance_chain.run(
                consultant_name=self.consultant_name,
                consultant_role=self.consultant_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_history="\n".join(self.conversation_history),
                conversation_stage=self.current_conversation_stage,
                conversation_type=self.conversation_type,
            )

        # Add agent's response to conversation history
        # Remove <END_OF_TURN> to keep it clean for display
        print(f"{self.consultant_name}: ", ai_message.replace("<END_OF_TURN>", ""))
        
        agent_name = self.consultant_name
        ai_message = agent_name + ": " + ai_message
        if "<END_OF_TURN>" not in ai_message:
            ai_message += " <END_OF_TURN>"
        self.conversation_history.append(ai_message)

        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True, **kwargs) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        conversation_utterance_chain = ConversationChain.from_llm(
            llm, verbose=verbose
        )

        Sales_agent_executor = None
        if kwargs.get("use_tools", False):
            tools = get_tools(llm)
            
            if tools:
                prompt = CustomPromptTemplateForTools(
                    template=SALES_AGENT_TOOLS_PROMPT,
                    tools_getter=lambda x: tools,
                    input_variables=[
                        "input",
                        "intermediate_steps",
                        "consultant_name",
                        "consultant_role",
                        "company_name",
                        "company_business",
                        "company_values",
                        "conversation_purpose",
                        "conversation_type",
                        "conversation_history",
                    ],
                )
                llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

                tool_names = [tool.name for tool in tools]

                output_parser = ConvoOutputParser(
                    ai_prefix=kwargs.get("consultant_name", "AI Assistant"), 
                    verbose=verbose
                )

                Sales_agent_with_tools = LLMSingleActionAgent(
                    llm_chain=llm_chain,
                    output_parser=output_parser,
                    stop=["\nObservation:"],
                    allowed_tools=tool_names,
                    verbose=verbose,
                )

                Sales_agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=Sales_agent_with_tools, tools=tools, verbose=verbose
                )
            else:
                print("No tools available. Running without tools.")

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            conversation_utterance_chain=conversation_utterance_chain,
            Sales_agent_executor=Sales_agent_executor,
            verbose=verbose,
            **kwargs,
        )
