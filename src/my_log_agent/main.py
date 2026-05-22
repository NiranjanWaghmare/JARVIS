#!/usr/bin/env python
import sys
import os
import warnings
from datetime import datetime
from my_log_agent.crew import MyLogAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    # Use environment goal
    topic = os.environ.get('JARVIS_GOAL', 'AI LLMs')
    inputs = {
        'topic': topic,
        'current_year': str(datetime.now().year)
    }

    try:
        # Standardize output for Windows console
        print("[JARVIS] Initializing agents...")
        crew = MyLogAgent().crew()
        print("[JARVIS] Starting analysis...")
        result = crew.kickoff(inputs=inputs)
        
        # Cleanly print result
        print("\n[FINAL REPORT]")
        print(str(result).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
        print("\n[JARVIS] Task completed successfully.")
        return 0
    except Exception as e:
        print(f"[ERROR] Jarvis encountered an issue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
    sys.exit(0)

# Keep other methods (train, replay, test) as standard
def train():
    try: MyLogAgent().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs={"topic":"AI"})
    except Exception as e: raise e

def replay():
    try: MyLogAgent().crew().replay(task_id=sys.argv[1])
    except Exception as e: raise e

def test():
    try: MyLogAgent().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs={"topic":"AI"})
    except Exception as e: raise e
