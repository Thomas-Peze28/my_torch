
NAME	=	my_torch_analyzer

GENERATOR	=	my_torch_generator

PYTHON	?=	python3

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
	$(PYTHON) -m pytest tests

dataset:
	$(PYTHON) -m scripts.prepare_dataset

train:
	$(PYTHON) -m scripts.train_network

.PHONY:	all $(NAME) $(GENERATOR) clean fclean re tests_run dataset train
