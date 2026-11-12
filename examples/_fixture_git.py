"""Git setup shared by the executable CPU examples."""

from fastci.repository import git


def init_fixture(repo, *, branch=None, name="FastCI fixture"):
    options = ("-b", branch) if branch else ()
    git(repo, "init", "-q", *options)
    git(repo, "config", "user.name", name)
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")


def commit_fixture(repo, message):
    git(repo, "add", ".")
    git(repo, "commit", "-qm", message)
