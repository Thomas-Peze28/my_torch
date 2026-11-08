
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
	$(PYTHON) -m scripts.prepare_dataset --mirror --swap-colors

train:
	mkdir -p benchmarks
	$(PYTHON) -m scripts.train_network --history benchmarks/pushed_network.json
	$(PYTHON) -m scripts.plot_history benchmarks/pushed_network.json --output benchmarks/pushed_network.png --title "my_torch_network.nn (chess_network.conf)"
	$(PYTHON) -m scripts.evaluate_network my_torch_network.nn --output benchmarks/pushed_network_confusion.png

benchmark:
	mkdir -p runs/plain runs/mirror benchmarks
	$(PYTHON) -m scripts.prepare_dataset --output runs/plain
	$(PYTHON) -m scripts.prepare_dataset --mirror --output runs/mirror
	for config in configs/benchmarks/*.conf; do \
		name=$$(basename $$config .conf); \
		$(PYTHON) -m scripts.train_network --config $$config --dataset runs/plain --output runs/$$name.nn --history benchmarks/$$name.json || exit 84; \
	done
	$(PYTHON) -m scripts.train_network --config configs/chess_network_256.conf --dataset runs/plain --output runs/augmentation_none.nn --history benchmarks/augmentation_none.json
	$(PYTHON) -m scripts.train_network --config configs/chess_network_256.conf --dataset runs/mirror --output runs/augmentation_mirror.nn --history benchmarks/augmentation_mirror.json
	$(PYTHON) -m scripts.train_network --config configs/chess_network_256.conf --output runs/augmentation_colors.nn --history benchmarks/augmentation_colors.json
	cp benchmarks/augmentation_colors.json benchmarks/size_256_64.json
	cp benchmarks/pushed_network.json benchmarks/size_512_256.json
	$(PYTHON) -m scripts.plot_history benchmarks/init_*.json --output benchmarks/init.png --title "Starting weights (769 -> 64 -> 5, SGD 0.1)"
	$(PYTHON) -m scripts.plot_history benchmarks/optimizer_*.json --output benchmarks/optimizer.png --title "Optimizer (769 -> 64 -> 5, He weights)"
	$(PYTHON) -m scripts.plot_history benchmarks/learning_rate_*.json --output benchmarks/learning_rate.png --title "Learning rate (769 -> 64 -> 5, Adam)"
	$(PYTHON) -m scripts.plot_history benchmarks/augmentation_*.json --output benchmarks/augmentation.png --title "More chessboards (769 -> 256 -> 64 -> 5, Adam)"
	$(PYTHON) -m scripts.plot_history benchmarks/size_*.json --output benchmarks/size.png --title "Network size (mirrored and swapped chessboards)"

.PHONY:	all $(NAME) $(GENERATOR) clean fclean re tests_run dataset train benchmark
