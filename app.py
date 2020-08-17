import os
os.environ["WANDB_SILENT"] = "true"

import json
import sys
import click

from github import Github

from label_bot import models



def init_models(ctx, param, value):
    global bot

    if not value or ctx.resilient_parsing:
        bot = models.Bot(use_head=False)
    else:
        bot = models.Bot(use_head=True)


@click.group()
@click.option("--use-head", "-h", is_flag=True, callback=init_models, expose_value=False)
def cli():
    pass


def get_token(file="token.json"):
    with open(file) as f:
        token = json.load(f)["token"]

    return token


def predict(title, body):
    return bot.predict(title, body)[0]


@cli.command("crawl-organization")
@click.option("--organization", "-O")
def run_on_org(organization):
    token = get_token()
    g = Github(token)

    root = g.get_organization(organization)

    for repo in root.get_repos():
        for issue in repo.get_issues():
            b_score, q_score, e_score = predict(issue.title, issue.body)
            results.append([repo.name, issue.number, b_score, q_score, e_score])

    return results


@cli.command("crawl-user")
@click.option("--user", "-U")
def run_on_user(user):
    token = get_token()
    g = Github(token)

    root = g.get_user(user)

    for repo in root.get_repos():
        for issue in repo.get_issues():
            b_score, q_score, e_score = predict(issue.title, issue.body)
            results.append([repo.name, issue.number, b_score, q_score, e_score])

    return results


@cli.command("crawl-repo")
@click.option("--repo", "-R")
def run_on_repo(repo):
    token = get_token()
    g = Github(token)

    repo = g.get_repo(repo)

    for issue in repo.get_issues():
        b_score, q_score, e_score = predict(issue.title, issue.body)
        results.append([repo.name, issue.number, b_score, q_score, e_score])

    return results


@cli.command("crawl-issue")
@click.option("--repo", "-R")
@click.option("--issue", "-I")
def run_on_issue(repo, issue):
    token = get_token()
    g = Github(token)

    repo = g.get_repo(repo)
    issue = repo.get_issue(number=issue)

    b_score, q_score, e_score = predict(issue.title, issue.body)
    results.append([repo.name, issue.number, b_score, q_score, e_score])

    return results


@cli.command("demo")
def demo():
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



if __name__ == "__main__":
    cli()