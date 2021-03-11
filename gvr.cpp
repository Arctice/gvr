#include <memory>
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>
#include <numeric>
#include <unordered_map>
#include <deque>
#include <cmath>

#include "geometry.h"
#include "vec3.h"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <fmt/core.h>

struct runtime_statistics& stats();

struct runtime_statistics {
    std::unordered_map<const char*, std::deque<double>> costs;

    struct perf_block {
        perf_block(const char* key) : key(key) {}
        ~perf_block() {
            auto delta = std::chrono::high_resolution_clock::now() - start;
            auto& store = stats().costs[key];
            store.push_back(delta.count() / 1000.0);
            if (store.size() > 10)
                store.pop_front();
        }

        const char* key;
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
    };

    auto time(const char* key) { return perf_block{key}; };

    double avg(const char* key) {
        auto& ts = costs[key];
        return std::accumulate(ts.begin(), ts.end(), 0., std::plus{}) /
               ts.size();
    }
};

runtime_statistics& stats() {
    static runtime_statistics self{};
    return self;
}

struct grid {
    grid(vec2i size) : size(size) {
        cells.resize(size.x * size.y);
        std::fill(cells.begin(), cells.end(), 0);
    }

    vec2i size;
    std::vector<float> cells;

    const float& at(vec2i v) const { return cells[v.y * size.x + v.x]; }
    float& at(vec2i v) { return cells[v.y * size.x + v.x]; }

    const float& operator[](vec2i v) const { return at(unwrap(v)); }
    float& operator[](vec2i v) { return at(unwrap(v)); }
    float& operator[](vec2f v) {
        return (*this)[vec2i{(int)std::floor(v.x), (int)std::floor(v.y)}];
    }
    const float& operator[](vec2f v) const {
        return (*this)[vec2i{(int)std::floor(v.x), (int)std::floor(v.y)}];
    }

private:
    vec2i unwrap(vec2i v) const {
        return vec2i{(v.x + size.x) % size.x, (v.y + size.y) % size.y};
    }
};

struct species {
    double speed;
    double rotation_angle;
    double sensor_angle;
    double sensor_distance;
    double deposit;
};

auto default_species() {
    static species self;
    self.speed = 5;
    self.rotation_angle = 18. * pi / 180.;
    self.sensor_angle = 12 * pi / 180.;
    self.sensor_distance = 20;
    self.deposit = 1.8;
    return &self;
}

constexpr double decay_factor = .68;
constexpr int sim_width = 2000;
constexpr int initial_population = 100000;

struct agent {
    vec2f position;
    vec2f direction;
    const species* species{default_species()};
};

struct simulation {
    simulation(vec2i size) : size(size), trails(size) {}
    vec2i size;
    grid trails;
    std::vector<agent> agents{};
};

namespace apx {

double cos(double x) {
    constexpr auto f4 = 2 * 3 * 4;
    constexpr auto f6 = f4 * 5 * 6;
    auto x2 = x * x;
    auto x4 = x2 * x2;
    auto x6 = x4 * x2;
    return 1. + x2 * (-1. / 2) + x4 * (1. / f4) + x6 * (-1. / f6);
}

double sin(double x) {
    constexpr auto f3 = 2 * 3;
    constexpr auto f5 = f3 * 4 * 5;
    constexpr auto f7 = f5 * 6 * 7;
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto x5 = x3 * x2;
    auto x7 = x5 * x2;
    return x + x3 * (-1. / f3) + x5 * (1. / f5) + x7 * (-1. / f7);
}

vec2f rotate(vec2f v, double a) {
    auto s = apx::sin(a);
    auto c = apx::cos(a);
    return {v.x * c - v.y * s, v.x * s + v.y * c};
}

}; // namespace apx

void move_agent(const grid& trails, agent& a) {
    auto fwd_sensor = a.direction * a.species->sensor_distance;
    auto left_sensor = apx::rotate(fwd_sensor, a.species->sensor_angle * -1);
    auto right_sensor = apx::rotate(fwd_sensor, a.species->sensor_angle);

    auto fwd_trail = trails[a.position + fwd_sensor];
    auto left_trail = trails[a.position + left_sensor];
    auto right_trail = trails[a.position + right_sensor];

    if (left_trail > fwd_trail and right_trail > fwd_trail) {
        auto side = (drand48() < .5) ? -1 : 1;
        a.direction =
            apx::rotate(a.direction, a.species->rotation_angle * side);
    }
    else if (left_trail > fwd_trail)
        a.direction = apx::rotate(a.direction, a.species->rotation_angle * -1);
    else if (right_trail > fwd_trail)
        a.direction = apx::rotate(a.direction, a.species->rotation_angle);

    a.position += a.direction * a.species->speed;
}

void advance_agents(simulation& state) {
#pragma omp parallel for schedule(static)
    for (auto& agent : state.agents)
        move_agent(state.trails, agent);
}

auto deposits(const simulation& state) {
    auto trails = state.trails;
#pragma omp parallel for schedule(static)
    for (auto& [pos, _, species] : state.agents)
        trails[pos] += species->deposit;
    return trails;
}

auto diffuse(const grid& trails) {
    auto new_trails = grid{trails.size};

    // #pragma omp parallel for schedule(static)
    for (int y = 1; y < trails.size.y - 1; ++y) {
        for (int x{1}; x < trails.size.x - 1; ++x) {
            float v{};
            for (int yk{y - 1}; yk < y + 2; ++yk)
                for (int xk{x - 1}; xk < x + 2; ++xk)
                    v += trails.at({xk, yk});
            new_trails.at(vec2i{x, y}) = v / 9.;
        }
    }

    for (int x{-1}; x < 1; ++x) {
        for (int y{0}; y < trails.size.y; ++y) {
            float v{};
            for (int yk{y - 1}; yk < y + 2; ++yk)
                for (int xk{x - 1}; xk < x + 2; ++xk)
                    v += trails[vec2i{xk, yk}];
            new_trails[vec2i{x, y}] = v / 9.;
        }
    }

    for (int y{-1}; y < 1; ++y) {
        for (int x{0}; x < trails.size.x; ++x) {
            float v{};
            for (int yk{y - 1}; yk < y + 2; ++yk)
                for (int xk{x - 1}; xk < x + 2; ++xk)
                    v += trails[vec2i{xk, yk}];
            new_trails[vec2i{x, y}] = v / 9.;
        }
    }

    return new_trails;
}

void decay(simulation& state) {
    for (auto& v : state.trails.cells)
        v *= decay_factor;

    for (auto& agent : state.agents) {
        while (agent.position.x < 0)
            agent.position.x += state.size.x;
        while (agent.position.y < 0)
            agent.position.y += state.size.y;
        while ((int)agent.position.x >= state.size.x)
            agent.position.x -= state.size.x;
        while ((int)agent.position.y >= state.size.y)
            agent.position.y -= state.size.y;
    }
}

auto tick(simulation state) {
    auto cost{stats().time("tick")};

    {
        auto cost{stats().time("move")};
        advance_agents(state);
    }

    {
        auto cost{stats().time("deposit")};
        state.trails = deposits(state);
    }

    {
        auto cost{stats().time("diffuse")};
        state.trails = diffuse(diffuse(diffuse(state.trails)));
    }

    {
        auto cost{stats().time("decay")};
        decay(state);
    }

    return state;
}

auto random_agent(vec2f size) {
    auto pos = vec2f{drand48() * size.x, drand48() * size.y};
    auto dir = vec2f{1.0, 0.}.rotated(drand48() * std::asin(-1) * 4);
    return agent{pos, dir};
}

struct renderer {
    renderer(vec2i resolution)
        : resolution(resolution), scale(.5f),
          window(sf::VideoMode{(unsigned int)(resolution.x * scale),
                               (unsigned int)(resolution.y * scale)},
                 "gvr") {
        sim_draw_buffer.resize(4 * resolution.y * resolution.x);
        std::fill(sim_draw_buffer.begin(), sim_draw_buffer.end(), 255);
    }

    void frame() {
        window.display();
        window.clear();
    }

    vec2i resolution;
    float scale;
    sf::RenderWindow window;
    std::vector<unsigned char> sim_draw_buffer;
};

void draw_simulation(renderer& context, const simulation& state) {
    auto resolution = state.size;

#pragma omp parallel for schedule(static)
    for (auto px = 0; px < state.trails.cells.size(); ++px) {
        auto v = state.trails.cells[px];
        v = std::sqrt(v * 10.f);
        v = 255 - std::min(v, 10.f) * 25.5f;
        context.sim_draw_buffer[4 * px] = v;
        context.sim_draw_buffer[4 * px + 1] = v;
        context.sim_draw_buffer[4 * px + 2] = v;
    }

    sf::Texture texture;
    texture.create((unsigned int)resolution.x, (unsigned int)resolution.y);
    texture.update(context.sim_draw_buffer.data());

    sf::Sprite view_sprite;
    view_sprite.setTexture(texture);
    view_sprite.setScale({context.scale, context.scale});
    context.window.draw(view_sprite);
}

int main() {
    srand48(std::chrono::nanoseconds(
                std::chrono::high_resolution_clock().now().time_since_epoch())
                .count());

    simulation state{{sim_width, sim_width}};

    for (int n{}; n < initial_population; ++n)
        state.agents.push_back(
            random_agent(vec2f{vec2i{state.size.x, state.size.y}}));

    renderer display{state.size};

    auto t_prev = std::chrono::high_resolution_clock::now();

    auto report_time = t_prev - std::chrono::seconds{1};
    while (display.window.isOpen()) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto delta = t_now - t_prev;
        {
            auto cost{stats().time("sleep")};
            std::this_thread::sleep_for(
                std::chrono::microseconds(1000000) / 60 - delta);
        }
        t_now = std::chrono::high_resolution_clock::now();
        delta = t_now - t_prev;
        t_prev = t_now;

        auto cost{stats().time("frame")};

        state = tick(state);

        {
            auto cost{stats().time("render")};
            draw_simulation(display, state);
            display.frame();
        }

        if (t_now - report_time > std::chrono::seconds{3}) {
            std::vector<std::pair<double, const char*>> best;
            for (const auto& [name, _] : stats().costs)
                best.push_back({stats().avg(name), name});

            std::sort(best.rbegin(), best.rend());
            for (const auto& [cost, name] : best) {
                auto percent = cost / (stats().avg("frame")) * 100.;
                if (cost > 1000)
                    fmt::print("{:6.1f}ms", cost / 1000.);
                else
                    fmt::print("{:6.1f}µs", cost);
                fmt::print(" {:<12} {:5.1f}%\n", name, percent);
            }
            fmt::print("\n");

            report_time = t_now;
        }

        sf::Event event;
        while (display.window.pollEvent(event))
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed &&
                 event.key.code == sf::Keyboard::Escape))
                display.window.close();
    }
}
