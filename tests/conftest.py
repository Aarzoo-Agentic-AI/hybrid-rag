import os

# Never send traces to LangSmith from the test suite. This runs before any test imports
# LangChain, so LangSmith's cached view of the environment starts out disabled.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
