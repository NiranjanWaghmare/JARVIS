import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import ScrapeWebsiteTool

@CrewBase
class MyLogAgent():
    """MyLogAgent crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(self):
        self.llm = LLM(model=os.environ.get("MODEL", "ollama/smollm2:1.7b"))

    @agent
    def researcher(self) -> Agent:
        config = self.agents_config['researcher']
        tools = [ScrapeWebsiteTool()] if config.get('use_tools') else []
        
        # Multi-Model Support: Use agent-specific LLM if provided
        agent_llm = self.llm
        if config.get('llm'):
            agent_llm = LLM(model=config['llm'])

        return Agent(
            config=config,
            llm=agent_llm,
            tools=tools,
            max_iter=config.get('max_iter', 10),
            reasoning=config.get('reasoning', False),
            inject_date=config.get('inject_date', True),
            verbose=False
        )

    @agent
    def reporting_analyst(self) -> Agent:
        config = self.agents_config['reporting_analyst']
        tools = [ScrapeWebsiteTool()] if config.get('use_tools') else []

        # Multi-Model Support: Use agent-specific LLM if provided
        agent_llm = self.llm
        if config.get('llm'):
            agent_llm = LLM(model=config['llm'])

        return Agent(
            config=config,
            llm=agent_llm,
            tools=tools,
            max_iter=config.get('max_iter', 10),
            reasoning=config.get('reasoning', False),
            inject_date=config.get('inject_date', True),
            verbose=False
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MyLogAgent crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False
        )
