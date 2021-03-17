flags = -std=c++17 -Wall -Wpedantic -ffast-math -march=native -fopenmp=libiomp5 -I./imgui

libs = 	-lomp5 -pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system -lfmt -lGL

imgui = imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
	imgui/imgui-SFML.cpp imgui/implot.cpp imgui/implot_items.cpp
src = gvr.cpp  ${imgui}
objs=$(src:.cpp=.o)

%.o : %.cpp
	clang++ ${flags} -c -o $@ $<

build: $(objs)
	clang++ ${flags} -o gvr ${objs} ${libs}

clean:
	rm gvr gvr.o imgui/*.o

core: flags += -O2
core: build

debug: flags += -fsanitize=undefined,address -g -fno-omit-frame-pointer
debug: build
