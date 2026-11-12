"""Isolated Git fixtures for tests run from a checkout or an installed wheel."""

from fastci.repository import git


def init_repo(repo, *, branch=None, name="FastCI fixture"):
    options = ("-b", branch) if branch else ()
    git(repo, "init", "-q", *options)
    git(repo, "config", "user.name", name)
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")


def commit(repo, message):
    git(repo, "add", ".")
    git(repo, "commit", "-qm", message)
