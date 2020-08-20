# Github-Issues-Classifier
# Copyright(C) 2018, 2019, 2020 Georgios (Giorgos) Karantonis
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
os.environ["WANDB_SILENT"] = "true"

import json
import sys
import click

import pandas as pd
from github import Github

from label_bot import models



def init_models(ctx, param, value):
    global BOT

    if not value or ctx.resilient_parsing:
        BOT = models.Bot(use_head=False)
    else:
        BOT = models.Bot(use_head=True)


def set_token(ctx, param, value):
    global token

    if not value or ctx.resilient_parsing:
        token = get_token()

        if not token:
            print("Exiting, no token found...")

        return
    
    token = value


def get_token(file="token_private.json"):
    with open(file) as f:
        token = json.load(f)["token"]

    return token


def predict(title, body):
    return BOT.predict(title, body)[0]


@click.group()
@click.option("--token", "-t", callback=set_token, expose_value=False)
@click.option("--use-head", "-h", is_flag=True, callback=init_models, expose_value=False)
@click.option("--threshold", "-th", default=.5, type=float)
@click.option("--apply-labels", "-l", is_flag=True)
def cli(threshold, apply_labels):
    global THRESHOLD, APPLY_LABELS

    THRESHOLD = threshold
    APPLY_LABELS = apply_labels

    pass


@cli.command("crawl-org")
@click.option("--organization", "-o")
def run_on_org(organization):
    results = pd.DataFrame(columns=["repo", "issue", "bug", "question", "enhancement"])

    token = get_token()
    g = Github(token)

    root = g.get_organization(organization)

    for repo in root.get_repos():
        for issue in repo.get_issues():
            b_score, q_score, e_score = predict(issue.title, issue.body)
            
            if APPLY_LABELS:
                for l, s in zip(("bug", "question", "enhancement"), (b_score, q_score, e_score)):
                    if s >= THRESHOLD:
                        issue.set_labels(l)
            
            results = results.append({"repo" : repo.name,
                                      "issue" : issue.number,
                                      "bug" : b_score,
                                      "question" : q_score,
                                      "enhancement" : e_score
                                    }, ignore_index=True)

    return results


@cli.command("crawl-user")
@click.option("--user", "-u")
def run_on_user(user):
    results = pd.DataFrame(columns=["repo", "issue", "bug", "question", "enhancement"])

    token = get_token()
    g = Github(token)

    root = g.get_user(user)

    for repo in root.get_repos():
        for issue in repo.get_issues():
            b_score, q_score, e_score = predict(issue.title, issue.body)

            if APPLY_LABELS:
                for l, s in zip(("bug", "question", "enhancement"), (b_score, q_score, e_score)):
                    if s >= THRESHOLD:
                        issue.set_labels(l)

            results = results.append({"repo" : repo.name,
                                      "issue" : issue.number,
                                      "bug" : b_score,
                                      "question" : q_score,
                                      "enhancement" : e_score
                                    }, ignore_index=True)

    return results


@cli.command("crawl-repo")
@click.option("--repo", "-r")
def run_on_repo(repo):
    results = pd.DataFrame(columns=["repo", "issue", "bug", "question", "enhancement"])

    token = get_token()
    g = Github(token)

    repo = g.get_repo(repo)

    for issue in repo.get_issues():
        b_score, q_score, e_score = predict(issue.title, issue.body)

        if APPLY_LABELS:
            for l, s in zip(("bug", "question", "enhancement"), (b_score, q_score, e_score)):
                if s >= THRESHOLD:
                    issue.set_labels(l)
        
        results = results.append({"repo" : repo.name,
                                  "issue" : issue.number,
                                  "bug" : b_score,
                                  "question" : q_score,
                                  "enhancement" : e_score
                                }, ignore_index=True)

    return results


@cli.command("crawl-issue")
@click.option("--repo", "-r")
@click.option("--issue", "-i")
def run_on_issue(repo, issue):
    results = pd.DataFrame(columns=["repo", "issue", "bug", "question", "enhancement"])

    token = get_token()
    g = Github(token)

    repo = g.get_repo(repo)
    issue = repo.get_issue(number=issue)

    b_score, q_score, e_score = predict(issue.title, issue.body)

    if APPLY_LABELS:
        for l, s in zip(("bug", "question", "enhancement"), (b_score, q_score, e_score)):
            if s >= THRESHOLD:
                issue.set_labels(l)

    results = results.append({"repo" : repo.name,
                              "issue" : issue.number,
                              "bug" : b_score,
                              "question" : q_score,
                              "enhancement" : e_score
                            }, ignore_index=True)

    return results


def demo():
    title = input("Title: ")
    body = input("Body: ")

    scores = predict(title, body)

    for kind, score in zip(("Bug", "Question", "Enhancement"), scores):
        print(f"{kind}: {score}")
    print()

    keep_going = input("Try another one? [y/n] ")

    while keep_going not in ("y", "n"):
      print("Type 'y' for YES or 'n' for NO")
      keep_going = input("Try another one? [y/n] ")

    if keep_going == "y":
        demo()
    else:
        sys.exit()


@cli.command("demo")
def start_demo():
    demo()



if __name__ == "__main__":
    cli()
