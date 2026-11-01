
NAME	=	my_torch_analyzer

GENERATOR	=	my_torch_generator

all:	$(NAME) $(GENERATOR)

$(NAME):
	chmod +x $(NAME)

$(GENERATOR):
	chmod +x $(GENERATOR)

clean:
	find . -name "__pycache__" -type d -prune -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache

fclean:	clean

re:	fclean all

tests_run:
	python3 -m pytest tests

.PHONY:	all $(NAME) $(GENERATOR) clean fclean re tests_run
