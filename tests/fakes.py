### Stand-ins used by the tests in place of a real Neo4j database and a real CrewAI crew.
### They record what the code asked for (queries, prompts, inputs), so tests can check it,
### and return prepared answers, so no database or LLM is ever contacted.

from types import SimpleNamespace


# --- Neo4j ---

class FakeResult:
    """Behaves like a Neo4j query result: can be iterated, or read with .data() / .single()."""
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def data(self):
        return list(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None


class FakeSession:
    def __init__(self, driver, database):
        self.driver = driver
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.driver.queries.append({"database": self.database, "query": query, "params": params})
        if self.driver.fail_on and self.driver.fail_on in query:
            raise Exception("simulated database failure")
        return FakeResult(self.driver.rows_for(query))

    def execute_read(self, work):
        return work(self)       # the session doubles as the read transaction


class FakeNeo4jDriver:
    """
    A pretend Neo4j connection.
      responses: {"text in query": [rows]}  -> rows returned for any query containing that text
      fail_on:   "text in query"            -> that query raises an error, like a failed write
    Every query is recorded in .queries; .closed shows whether the code closed the connection.
    """
    def __init__(self, responses=None, fail_on=None):
        self.responses = responses or {}
        self.fail_on = fail_on
        self.queries = []
        self.closed = False

    def rows_for(self, query):
        for text, rows in self.responses.items():
            if text in query:
                return rows
        return []

    def session(self, database=None):
        return FakeSession(self, database)

    def close(self):
        self.closed = True

    def queries_containing(self, text):
        return [q for q in self.queries if text in q["query"]]


def use_fake_neo4j(monkeypatch, module, **driver_options):
    """Makes module's GraphDatabase.driver(...) return a FakeNeo4jDriver, and returns that driver."""
    driver = FakeNeo4jDriver(**driver_options)
    monkeypatch.setattr(module.GraphDatabase, "driver", lambda *args, **kwargs: driver)
    return driver


# --- CrewAI ---

def fake_crew_class(raw_result="fake answer"):
    """
    Returns (FakeCrew, created): a class to put in place of crewai.Crew, and a list that collects
    every crew the code builds, so tests can inspect its agents, tasks and kickoff inputs.
    """
    created = []

    class FakeCrew:
        def __init__(self, agents=None, tasks=None, **kwargs):
            self.agents = agents
            self.tasks = tasks
            self.options = kwargs
            self.inputs = None
            created.append(self)

        def kickoff(self, inputs=None):
            self.inputs = inputs
            return SimpleNamespace(raw=raw_result)

    return FakeCrew, created
