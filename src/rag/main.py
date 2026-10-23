from dotenv import load_dotenv

load_dotenv()

from langchain_core import __version__ as core_version
from langgraph.version import __version__ as lg_version
from langchain_openrouter import ChatOpenRouter

print(f"lc cor vers {core_version}")
print(f"lg vers {lg_version}")


def main():
    llm = ChatOpenRouter(model="nvidia/nemotron-3-ultra-550b-a55b:free", temperature=0)
    response = llm.invoke("say setup complete in one word")

    print(response)


if __name__ == "__main__":
    main()
