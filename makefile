core:
	clang++ -std=c++17 -O3 -march=native -lomp5 \
	-Wall -Wpedantic \
	gvr.cpp -o gvr \
	-pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system -lfmt
debug:
	clang++ -std=c++17 -O1 -march=native -lomp5 \
	-fsanitize=undefined,address -g -fno-omit-frame-pointer \
	-Wall -Wpedantic \
	gvr.cpp -o gvr \
	-pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system -lfmt
