#include <memory>
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>

#include "geometry.h"
#include "vec3.h"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <fmt/core.h>

vec3<unsigned char> rgb_light(vec3f light) {
    auto clamp = [](auto c) { return std::min(1., std::max(0., c)); };
    light = {clamp(light.x), clamp(light.y), clamp(light.z)};
    return vec3<unsigned char>(light * 255.);
}

struct grid {
    grid(vec2i size) : size(size) {
        cells.resize(size.x * size.y);
        std::fill(cells.begin(), cells.end(), 0);
    }

    vec2i size;
    std::vector<float> cells;

    const float& operator[](vec2i v) const {
        v = unwrap(v);
        return cells[v.y * size.x + v.x];
    }
    float& operator[](vec2i v) {
        v = unwrap(v);
        return cells[v.y * size.x + v.x];
    }
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

struct species{
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
constexpr int sim_width = 800;
constexpr int initial_population = 40000;

struct agent {
    vec2f position;
    vec2f direction;
    const species* species{default_species()};
};

agent move_agent(const grid& trails, agent a) {
    a.direction =
        [&]() {
            auto pos = a.position;
            auto dir = a.direction;
            auto fwd = dir * a.species->sensor_distance;
            auto sa = a.species->sensor_angle;
            auto [left, right] = std::tuple{fwd.rotated(sa), fwd.rotated(-sa)};
            auto [s_left, s_fwd, s_right] = std::tuple{
                trails[pos + left], trails[pos + fwd], trails[pos + right]};

            auto r = a.species->rotation_angle;
            auto r_left = dir.rotated(r);
            auto r_right = dir.rotated(-r);

            if (s_fwd < s_left and s_fwd > s_right)
                return r_left;
            if (s_fwd > s_left and s_fwd < s_right)
                return r_right;
            if (s_fwd < s_left and s_fwd < s_right)
                return drand48() < .5 ? r_right : r_left;
            return dir;
        }()
            .normalized();
    a.position += a.direction * a.species->speed;
    return a;
}

auto advance_agents(const grid& trails, std::vector<agent> agents) {
#pragma omp for
    for (auto& agent : agents)
        agent = move_agent(trails, agent);
    return agents;
}

auto spread(grid trails, const std::vector<agent> agents) {
    for (auto& [pos, _, species] : agents)
        trails[pos] += species->deposit;
    return trails;
}

auto diffuse(const grid& trails) {
    auto next = grid{trails.size};
    std::array<vec2i, 9> kv = {
        vec2i{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {0, 0},
        {1, 0},        {-1, 1}, {0, 1},  {1, 1},
    };

    for (int y{1}; y < trails.size.y - 1; ++y) {
        for (int x{1}; x < trails.size.x - 1; ++x) {
            float v{};
            auto p = vec2i{x, y};
            for (int k{}; k < 9; ++k)
                v += trails[p + kv[k]];
            next[p] = v / 9.;
        }
    }
    return next;
}

auto decay(grid trails) {
    for (int y{1}; y < trails.size.y - 1; ++y) {
        for (int x{1}; x < trails.size.x - 1; ++x) {
            trails[vec2i{x, y}] *= decay_factor;
        }
    }
    return trails;
}

auto tick(grid trails, std::vector<agent> agents) {
    trails = spread(trails, agents);
    agents = advance_agents(trails, agents);
    trails = diffuse(diffuse(diffuse(trails)));
    trails = decay(trails);
    for(auto& agent: agents){
        while(agent.position.x < 0) agent.position.x += trails.size.x;
        while(agent.position.y < 0) agent.position.y += trails.size.y;
        while(agent.position.x > trails.size.x) agent.position.x -= trails.size.x;
        while(agent.position.y > trails.size.y) agent.position.y -= trails.size.y;
    }
    return std::pair{trails, agents};
}

auto random_agent(vec2f size) {
    auto pos = vec2f{drand48() * size.x, drand48() * size.y};
    auto dir = vec2f{1.0, 0.}.rotated(drand48() * std::asin(-1) * 4);
    return agent{pos, dir};
}

void export_img(const grid& trails, const std::vector<agent>&,
                std::string filename) {
    auto resolution = trails.size;
    std::vector<unsigned char> out;
    out.resize(4 * resolution.y * resolution.x);

    for (int y = 0; y < resolution.y; ++y) {
        for (int x = 0; x < resolution.x; ++x) {
            auto v = trails[vec2i{x, y}] * 10.;
            v = v < 0.1 ? 0. : std::log(v) / std::log(2.);
            v = 255 - std::clamp(0., v, 10.) * 25.5;
            auto [r, g, b] = std::tuple<int, int, int>{v, v, v};
            out[4 * (y * resolution.x + x)] = r;
            out[4 * (y * resolution.x + x) + 1] = g;
            out[4 * (y * resolution.x + x) + 2] = b;
            out[4 * (y * resolution.x + x) + 3] = 255;
        }
    }

    sf::Image img;
    img.create((unsigned int)(resolution.x), (unsigned int)(resolution.y),
               out.data());
    img.saveToFile(filename);
}

int main() {
    srand48(std::chrono::nanoseconds(
                std::chrono::high_resolution_clock().now().time_since_epoch())
                .count());
    auto trails = grid({sim_width, sim_width});
    auto agents = std::vector<agent>{};
    for (int n{}; n < initial_population; ++n)
        agents.push_back(
            random_agent(vec2f{(double)trails.size.x, (double)trails.size.y}));

    auto resolution = trails.size;
    float scaling = sim_width / float(trails.size.x);
    std::vector<unsigned char> out;
    out.resize(4 * resolution.y * resolution.x);

    sf::RenderWindow window{
        sf::VideoMode{(unsigned int)(resolution.x * scaling),
                      (unsigned int)(resolution.y * scaling)},
        "gvr"};

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int n{}; n < 10000; ++n) {
        std::tie(trails, agents) = tick(trails, agents);

        for (int y = 0; y < resolution.y; ++y) {
            for (int x = 0; x < resolution.x; ++x) {
                auto v = trails[vec2i{x, y}] * 10.;
                v = v < 0.1 ? 0. : std::log(v) / std::log(2);
                v = 255 - std::ceil(std::clamp(v, 0., 10.) * 25.5);
                auto [r, g, b] = std::tuple<int, int, int>{v, v, v};
                out[4 * (y * resolution.x + x)] = r;
                out[4 * (y * resolution.x + x) + 1] = g;
                out[4 * (y * resolution.x + x) + 2] = b;
                out[4 * (y * resolution.x + x) + 3] = 255;
            }
        }

        sf::Texture texture;
        texture.create((unsigned int)resolution.x, (unsigned int)resolution.y);
        texture.update(out.data());
        sf::Sprite view_sprite;
        view_sprite.setTexture(texture);
        view_sprite.setScale({scaling, scaling});

        window.clear();
        window.draw(view_sprite);
        window.display();

        auto now = std::chrono::high_resolution_clock::now();
        auto delta = now  - t0;
        t0 = now;

        std::this_thread::sleep_for(std::chrono::nanoseconds(1000000000 / 60) -
                                    delta);

        sf::Event event;
        while (window.pollEvent(event))
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed &&
                 event.key.code == sf::Keyboard::Escape))
                window.close();

        if (not window.isOpen())
            break;

        // export_img(trails, agents, fmt::format("vid/f{:0>3}.png", n));
    }
}

// The model postulated by Jones employs both an agent-based layer (the data
// map) and a continuum-based layer (the trail map). The data map consists of
// many particles, while the trail map consists of a 2D grid of intensities
// (similar to a pixel-based image). The data and trail map in turn affect each
// other; the particles of the data map deposit material onto the trail map,
// while those same particles sense values from the trail map in order to
// determine aspects of their locomotion.

// Each particle in the simulation has a heading angle, a location, and three
// sensors (front left, front, front right). The sensor readings effect the
// heading of the particle, causing it to rotate left or right (or stay facing
// the same direction). The trail map undergoes a diffusion and decay process
// every simulation step. A simple 3-by-3 mean filter is applied to simulate
// diffusion of the particle trail, and then a multiplicative decay factor is
// applied to simulate trail dissipation over time. The diagram below describes
// the six sub-steps of a simulation tick.

// Many of the parameters of this simulation are configurable, including sensor
// distance, sensor size, sensor angle, step size, rotation angle, deposition
// amount, decay factor, deposit size, diffuse size, decay factor, etc. For a
// more detailed description check out the original paper.

// There are several substantial differences between the model as described by
// Jones and my implementation. In Jones’s original paper there is a collision
// detection step that ensures that there is at most one particle in each grid
// square. For my implementations I usually ignored this step, preferring the
// patterns that arose without it. However, this step is crucial for exact
// mimicry of the behavior of Physarum polycephalum, as it approximates a sort
// of conservation of matter. Also (conveniently) this collision detection
// removes any sort of sequential dependence, allowing for increased
// computational parallelism.

// std::cerr << (std::chrono::high_resolution_clock::now() - t0).count() /
//                  1000000.
//           << "ms\n";

// (define (box-blur line)
//   (define kernel 3)
//   (define radius (fx/ kernel 2))
//   (define width (vector-length line))
//   (let ([line* (make-vector width 0.)])
//     (let slide ([r (sum (map (partial vector-ref line) (iota kernel)))]
//                 [x 0])
//       (vector-set! line* (fx+ x radius) (fl/ r (inexact kernel)))
//       (if (>= x (fx- width kernel)) line*
//           (slide (fl+ r (vector-ref line (fx+ x kernel))
//                     (fl- (vector-ref line x)))
//                  (inc x))))))

// (define (transpose M)
//   (let* ([size (vector-length M)]
//          [N (make-grid size)])
//     (do ([y 0 (inc y)]) ((>= y size) N)
//       (let ([row (vector-ref M y)])
//         (do ([x 0 (inc x)]) ((>= x size))
//           (vector-set! (vector-ref N x) y
//                        (vector-ref row x)))))))

// (define (diffuse trails)
//   (let ([gauss (compose box-blur box-blur box-blur)])
//     (transpose
//      (list->vector
//       (map gauss
//            ((compose vector->list transpose list->vector)
//             (map gauss (vector->list trails))))))))
