import sys
import click

from label_bot import models



@click.group()
def cli():
	pass


def predict(title, body):
	return bot.predict(title, body)[0]	


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


@cli.command("run-demo")
@click.option("--use-head", "-h", default=True, type=bool)
def start_demo(use_head):
	global bot
	bot = models.Bot(use_head=use_head)

	demo()


if __name__ == "__main__":
	cli()
	