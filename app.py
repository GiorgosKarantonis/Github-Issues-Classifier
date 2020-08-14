import os
os.environ["WANDB_SILENT"] = "true"

import json
import sys
import click

from github import Github

from label_bot import models



@click.group()
def cli():
    pass


def deploy(token, user=None, organization=None, repo=None, issue=None):
    g = Github(token)
    titles, bodies = [], []
    results = []

    if user or organization:
        if user:
            root = g.get_user(user)
        elif organization:
            root = g.get_organization(organization)

        for repo in root.get_repos():
            for issue in repo.get_issues():
                b_score, q_score, e_score = predict(issue.title, issue.body)

                results.append([repo.name, issue.number, b_score, q_score, e_score])
    elif repo:
        repo = g.get_repo(repo)
        
        if issue:
            issue = repo.get_issue(number=issue)
            
            results.append([repo.name, issue.number, b_score, q_score, e_score])
        else:
            for issue in repo.get_issues():
                b_score, q_score, e_score = predict(issue.title, issue.body)

                results.append([repo.name, issue.number, b_score, q_score, e_score])

    return results


def predict(title, body):
    return bot.predict(title, body)


def demo():
    from_issue = input("Run from repo issue? [y/n] ")

    if from_issue == "y":
        repo = input("Repo: ")
        issue = int(input("Issue number: "))

        with open("token.json") as f:
            token = json.load(f)['token']

        g = Github(token)
        issue = g.get_repo(repo).get_issue(number=issue)

        title, body = issue.title, issue.body
    else:
        title = input("Title: ")
        body = input("Body: ")

    b_score, q_score, e_score = predict(title, body)

    print(f"Bug: {b_score}")
    print(f"Question: {q_score}")
    print(f"Enhancement: {e_score}")
    print()

    keep_going = input("Try another one? [y/n] ")

    assert keep_going in ["y", "n"]
    if keep_going == "y":
        demo()
    else:
        sys.exit()


@cli.command("run-demo")
@click.option("--use-head", "-h", default=True, type=bool)
def start_demo(use_head):
    # global bot
    # bot = models.Bot(use_head=use_head)

    demo()


if __name__ == "__main__":
    cli()