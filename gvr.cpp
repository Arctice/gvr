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

#include "imgui/imgui.h"
#include "imgui/imgui-SFML.h"

struct runtime_statistics& stats();

struct runtime_statistics {
    std::unordered_map<const char*, std::deque<double>> costs;

    struct perf_block {
        perf_block(const char* key) : key(key) {}
        ~perf_block()
        {
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

    double avg(const char* key)
    {
        auto& ts = costs[key];
        return std::accumulate(ts.begin(), ts.end(), 0., std::plus{}) /
               ts.size();
    }
};

runtime_statistics& stats()
{
    static runtime_statistics self{};
    return self;
}

template <typename T = float>
struct grid {
    grid(vec2i size) : size(size)
    {
        cells.resize(size.x * size.y);
        std::fill(cells.begin(), cells.end(), T{});
    }

    vec2i size;
    std::vector<T> cells;

    const T& at(vec2i v) const { return cells[v.y * size.x + v.x]; }
    T& at(vec2i v) { return cells[v.y * size.x + v.x]; }

    const T& operator[](vec2i v) const { return at(unwrap(v)); }
    T& operator[](vec2i v) { return at(unwrap(v)); }
    T& operator[](vec2f v)
    {
        return (*this)[vec2i{(int)std::floor(v.x), (int)std::floor(v.y)}];
    }
    const T& operator[](vec2f v) const
    {
        return (*this)[vec2i{(int)std::floor(v.x), (int)std::floor(v.y)}];
    }

// private:
    vec2i unwrap(vec2i v) const
    {
        return vec2i{(v.x + size.x) % size.x, (v.y + size.y) % size.y};
    }
};

struct species {
    float speed{3.5f};
    float rotation_angle{24*pi/180.f};
    float sensor_angle{36*pi/180.f};
    float sensor_distance{18};
    float collision_rate{.8};
    float deposit{1};
    float decay_factor{.88};
    float color[3] = {1, 1, 1};
};

struct engine_params {
    static constexpr int sim_width{1000};
    int initial_population{0};
    int diffuse_passes{1};
    bool decay_cutoff{false};
    std::unordered_map<int, species> families{};
} config;

species* selected_species{nullptr};

struct agent {
    vec2f position;
    vec2f direction;
};

enum class collision : char { unoccupied, occupied, moved };

struct population {
    population(vec2i size, species* species)
        : size(size), species(species), trails(size), collisions{size}
    {}
    vec2i size;
    species* species;
    grid<float> trails;
    std::vector<agent> agents{};
    grid<collision> collisions;
};

struct simulation{
    vec2i size;
    std::vector<population> families{};
};

namespace apx {

double cos(double x)
{
    constexpr auto f4 = 2 * 3 * 4;
    constexpr auto f6 = f4 * 5 * 6;
    auto x2 = x * x;
    auto x4 = x2 * x2;
    auto x6 = x4 * x2;
    return 1. + x2 * (-1. / 2) + x4 * (1. / f4) + x6 * (-1. / f6);
}

double sin(double x)
{
    constexpr auto f3 = 2 * 3;
    constexpr auto f5 = f3 * 4 * 5;
    constexpr auto f7 = f5 * 6 * 7;
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto x5 = x3 * x2;
    auto x7 = x5 * x2;
    return x + x3 * (-1. / f3) + x5 * (1. / f5) + x7 * (-1. / f7);
}

vec2f rotate(vec2f v, double a)
{
    auto s = apx::sin(a);
    auto c = apx::cos(a);
    return {v.x * c - v.y * s, v.x * s + v.y * c};
}

}; // namespace apx

auto random_direction()
{
    return vec2f{1.0, 0.}.rotated(drand48() * std::asin(-1) * 4);
}

auto random_agent(vec2f size)
{
    auto pos = vec2f{drand48() * size.x, drand48() * size.y};
    return agent{pos, random_direction()};
}

void move_agent(population& state, agent& a)
{
    auto fwd_sensor = a.direction * state.species->sensor_distance;
    auto left_sensor =
        apx::rotate(fwd_sensor, state.species->sensor_angle * -1);
    auto right_sensor = apx::rotate(fwd_sensor, state.species->sensor_angle);

    auto fwd_trail = state.trails[a.position + fwd_sensor];
    auto left_trail = state.trails[a.position + left_sensor];
    auto right_trail = state.trails[a.position + right_sensor];

    if (left_trail > fwd_trail and right_trail > fwd_trail) {
        auto side = (drand48() < .5) ? -1 : 1;
        a.direction =
            apx::rotate(a.direction, state.species->rotation_angle * side);
    }
    else if (left_trail > fwd_trail)
        a.direction =
            apx::rotate(a.direction, state.species->rotation_angle * -1);
    else if (right_trail > fwd_trail)
        a.direction = apx::rotate(a.direction, state.species->rotation_angle);

    auto move = a.position + a.direction * state.species->speed;
    auto collision = state.collisions[move] != collision::unoccupied &&
                     (drand48() < state.species->collision_rate *
                                      state.species->collision_rate);
    if (collision) {
        a.direction = random_direction();
    }
    else {
        state.collisions[move] = collision::moved;
        a.position = move;
    }
}

void advance_agents(population& state)
{
    std::random_shuffle(state.agents.begin(), state.agents.end());
    std::fill(state.collisions.cells.begin(), state.collisions.cells.end(),
              collision::unoccupied);
    for (auto& agent : state.agents) {
        state.collisions[agent.position] = collision::occupied;
    }

    for (auto& agent : state.agents) {
        move_agent(state, agent);
    }
}

auto deposits(const population& state)
{
    auto trails = state.trails;
    // #pragma omp parallel for schedule(static)
    for (auto& [pos, _] : state.agents) {
        if (state.collisions[pos] == collision::moved)
            trails[pos] += state.species->deposit;
    }
    return trails;
}

auto diffuse(const grid<float>& trails)
{
    auto new_trails = grid{trails.size};

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

void decay(population& state)
{
    for (auto& v : state.trails.cells) {
        v *= state.species->decay_factor;
        if (config.decay_cutoff and v < .0001)
            v = 0;
    }

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

void consolidate(simulation& state)
{
    grid<float> trails{state.size};
    for (auto& family : state.families) {
        for (auto n{0}; n < family.trails.cells.size(); ++n)
            trails.cells[n] += family.trails.cells[n];
    }

    for (auto& family : state.families) {
        for (auto n{0}; n < family.trails.cells.size(); ++n)
            if (trails.cells[n] > 0)
                family.trails.cells[n] *=
                    family.trails.cells[n] / trails.cells[n];
    }
}

auto tick(population state)
{
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
        for (int pass{}; pass < config.diffuse_passes; ++pass)
            state.trails = diffuse(state.trails);
    }

    {
        auto cost{stats().time("decay")};
        decay(state);
    }

    return state;
}

struct renderer {
    renderer(vec2i size)
        : scale(1.f), resolution(size * scale),
          window(sf::VideoMode{(unsigned int)(resolution.x),
                               (unsigned int)(resolution.y)},
                 "fff")
    {
        sim_draw_buffer.resize(4 * size.y * size.x);
        std::fill(sim_draw_buffer.begin(), sim_draw_buffer.end(), 255);
    }

    void frame()
    {
        window.display();
        window.clear();
    }

    float scale;
    vec2i resolution;
    sf::RenderWindow window;
    std::vector<unsigned char> sim_draw_buffer;
};

void draw_simulation(renderer& context, const simulation& sim)
{
    for (auto px = 0; px < sim.size.x * sim.size.y; ++px) {
        vec3<float> c{};
        float v{};

        for (auto& family : sim.families) {
            auto [r, g, b] = family.species->color;
            c += vec3<float>{r, g, b} * family.trails.cells[px];
            v += family.trails.cells[px];
        }

        c /= v;

        v /= sim.families.size();
        v = std::sqrt(v * 100.f);
        v = std::min(v, 10.f) * 25.5f;

        c *= v;

        context.sim_draw_buffer[4 * px] = c.x;
        context.sim_draw_buffer[4 * px + 1] = c.y;
        context.sim_draw_buffer[4 * px + 2] = c.z;
    }

    sf::Texture texture;
    texture.create((unsigned int)sim.size.x, (unsigned int)sim.size.y);
    texture.update(context.sim_draw_buffer.data());

    sf::Sprite view_sprite;
    view_sprite.setTexture(texture);
    view_sprite.setScale({context.scale, context.scale});
    context.window.draw(view_sprite);
}

int main()
{
    srand48(std::chrono::nanoseconds(
                std::chrono::high_resolution_clock().now().time_since_epoch())
                .count());

    auto initial_species = species{};
    initial_species.color[0] = .2;
    initial_species.color[1] = .2;
    initial_species.color[2] = 1;
    config.families[config.families.size()] = initial_species;
    selected_species = &config.families[0];

    simulation state{{config.sim_width, config.sim_width}};
    state.families.emplace_back(state.size, selected_species);

    for (auto& p : state.families) {
        for (int n{}; n < config.initial_population; ++n)
            p.agents.push_back(random_agent(vec2f{state.size}));
    }

    renderer display{state.size};

    auto t_prev = std::chrono::high_resolution_clock::now();

    ImGui::SFML::Init(display.window);

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

        std::thread render_thread{[&display, state]() {
            auto cost{stats().time("render")};
            draw_simulation(display, state);
        }};

        for (auto& population : state.families)
            population = tick(population);

        consolidate(state);


        render_thread.join();

        {
            auto cost{stats().time("gui")};
            ImGui::SFML::Update(display.window,
                                sf::milliseconds(delta.count() / 1000.));

            ImGui::Begin("Config");

            static auto last_selection{0};
            if (ImGui::BeginCombo("Species",
                                  std::to_string(last_selection).c_str())) {
                for (int n{0}; n < config.families.size(); ++n) {
                    bool selected = selected_species == &config.families[n];
                    if (ImGui::Selectable(std::to_string(n).c_str(),
                                          selected)) {
                        selected_species = &config.families[n];
                        last_selection = n;
                        break;
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Checkbox("Cutoff", &config.decay_cutoff);

            if(ImGui::Button("Randomize")){
                for(auto& population: state.families){
                    population.trails = grid<float>{population.size};
                    for (auto& a : population.agents)
                        a = random_agent(vec2f{state.size});
                }
            }

            ImGui::SliderInt("Diffusion passes", &config.diffuse_passes, 0, 5);

            ImGui::SliderFloat("Decay", &selected_species->decay_factor, 0.01,
                               0.999);
            ImGui::SliderFloat("Speed", &selected_species->speed, 1.6, 40);

            ImGui::SliderAngle("Rotation angle",
                               &selected_species->rotation_angle, 1.f, 89.f,
                               "%.0f deg", 0);
            ImGui::SliderAngle("Sensor angle", &selected_species->sensor_angle,
                               1.f, 89.f, "%.0f deg", 0);

            ImGui::SliderFloat("Sensor Distance",
                               &selected_species->sensor_distance, 1, 100);

            ImGui::SliderFloat("Deposit", &selected_species->deposit, 0.1, 10);

            ImGui::SliderFloat("Collision rate",
                               &selected_species->collision_rate, 0.0, 1.0);

            ImGui::ColorEdit3("Color", selected_species->color);

            if (ImGui::Button("New species")) {
                state.families.emplace_back(
                    state.size,
                    &(config.families[config.families.size()] = species{}));
            }

            population* selected_family = &state.families.front();
            for (int n{}; n < state.families.size(); ++n) {
                if (state.families[n].species == selected_species) {
                    selected_family = &state.families[n];
                    break;
                }
            }

            if (ImGui::IsMouseDown(0) and !ImGui::GetIO().WantCaptureMouse) {
                auto [x, y] = ImGui::GetMousePos();
                auto inbounds = x > 0 && y > 0 && x < display.resolution.x &&
                                y < display.resolution.y;
                if (inbounds) {
                    auto a = random_agent({});
                    a.position = {x / display.scale, y / display.scale};
                    for (int n{}; n < 100; ++n)
                        selected_family->agents.push_back(a);
                }
            }

            if (ImGui::IsMouseDown(1) and !ImGui::GetIO().WantCaptureMouse) {
                auto count = selected_family->agents.size() / 10 + 1;
                for (int n{}; n < count; ++n)
                    if (!selected_family->agents.empty())
                        selected_family->agents.pop_back();
            }

            ImGui::End();
            ImGui::EndFrame();

            ImGui::SFML::Render(display.window);
        }

        display.frame();

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
        while (display.window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed &&
                 event.key.code == sf::Keyboard::Escape)) {
                display.window.close();
                ImGui::SFML::Shutdown();
                break;
            }

            ImGui::SFML::ProcessEvent(event);
        }
    }
}
