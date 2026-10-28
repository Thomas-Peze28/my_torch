
NAME	=	my_torch_analyzer

all:	$(NAME)

$(NAME):
	chmod +x $(NAME)

clean:
	find . -name "__pycache__" -type d -prune -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache

fclean:	clean

re:	fclean all

tests_run:
	python3 -m pytest tests

.PHONY:	all $(NAME) clean fclean re tests_run
