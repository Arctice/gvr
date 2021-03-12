dev:
	clang++ -std=c++17 -O0 -ffast-math -march=native -fopenmp=libiomp5 -lomp5 \
	-Wall -Wpedantic -fno-omit-frame-pointer \
	 -I./imgui imgui/*.cpp \
	gvr.cpp -o gvr \
	-pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system -lfmt -lGL
core:
	clang++ -std=c++17 -O2 -ffast-math -march=native -flto -fopenmp=libiomp5 -lomp5 \
	 -I./imgui imgui/*.cpp \
	gvr.cpp -o gvr \
	-pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system -lfmt -lGL
debug:
	clang++ -std=c++17 -O1 -march=native -lomp5 \
	-fsanitize=undefined,address -g -fno-omit-frame-pointer \
	-Wall -Wpedantic \
	gvr.cpp -o gvr \
	-pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system -lfmt
