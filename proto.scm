(library-directories "~/dev/core")
(import (scheme) (core) (random) (thread))
;; (random-seed (time-nanosecond (current-time 'time-utc)))
(random-seed 1)

(define-record-type vec2 (fields (immutable x) (immutable y)))

(define (vec2+ a b) (make-vec2 (+ (vec2-x a) (vec2-x b))
                               (+ (vec2-y a) (vec2-y b))))
(define (vec2- a b) (make-vec2 (- (vec2-x a) (vec2-x b))
                               (- (vec2-y a) (vec2-y b))))
(define (vec2-rotate v a)
  (let ([x (vec2-x v)] [y (vec2-y v)] [s (sin a)] [c (cos a)])
    (make-vec2 (- (* x c) (* y s)) (+ (* x s) (* y c)))))
(define (quantize-vec2 vec)
  (make-vec2 (+ (exact (floor (vec2-x vec))))
             (+ (exact (floor (vec2-y vec))))))

(define make-grid
  (case-lambda
   [(x) (make-grid x x)]
   [(x y) (vector-map (λ _ -> (make-vector x 0.)) (make-vector y))]))
(define (grid-center grid) (let ([size (floor (/ (vector-length grid) 2))])
                             (make-vec2 size size)))
(define (grid-ref grid v)
  (vector-ref (vector-ref grid (vec2-y v)) (vec2-x v)))
(define (grid-set! grid v x)
  (vector-set! (vector-ref grid (vec2-y v)) (vec2-x v) x))
(define (grid-sample grid position)
  (grid-ref grid (vec2+ (grid-center grid)
                        (quantize-vec2 position))))

(define sensor-angle 0.52)

(define-record-type agent
  (fields position direction))
(define (agent-sensors agent) 
  (map (λ angle -> (vec2+ (vec2-rotate (agent-direction agent) angle)
                          (agent-position agent)))
       (list 0 (- sensor-angle) sensor-angle)))

(define (agent-update trails agent)
  (let* ([sensors (agent-sensors agent)]
         [probes (map (partial grid-sample trails) sensors)])
    (apply
     (λ F L R ->
        (let ([fwd (make-agent (car sensors) (agent-direction agent))]
              [lft (make-agent (cadr sensors)
                               (vec2-rotate (agent-direction agent)
                                            (- sensor-angle)))]
              [rgt (make-agent (caddr sensors)
                               (vec2-rotate (agent-direction agent)
                                            sensor-angle))])
          (cond [(and (> F L) (> F R)) fwd]
                [(and (< F L) (< F R)) (choice (list lft rgt))]
                [(and (< F L) (> F R)) lft]
                [(and (> F L) (< F R)) rgt]
                [else fwd])))
     probes)))

(define (new-trails trails agents)
  (let ([c (grid-center trails)]
        [g (vector-map
            (λ y -> (vector-copy (vector-ref trails y)))
            (list->vector (iota (vector-length trails))))])
    (for-each
     (λ a -> (let ([cell (vec2+ c (quantize-vec2 (agent-position a)))])
               (grid-set! g cell (+ .9 (grid-ref g cell)))))
     agents)
    g))

;; (define (diffuse trails)
;;   (define filter
;;     (map pair (map (partial apply pair)
;;                    '((-1 -1) (0 -1) (1 -1)
;;                      (-1 0)  (0 0)  (1 0)
;;                      (-1 1)  (0 1)  (1 1)))
;;          (list .035 .05 .035
;;                .05  .65 .05
;;                .035 .05 .035)))

;;   (define size (vector-length trails))
;;   (let ([trails* (make-grid size)])
;;     (map (λ y ->
;;             (do ([x 1 (inc x)])
;;                 ((< (- size 2) x))
;;               (vector-set!
;;                (vector-ref trails* y) x
;;                (fold-left
;;                 (λ sum k ->
;;                    (let* ([v (head k)]
;;                           [row (vector-ref trails (fx+ y (tail v)))]
;;                           [inflow (vector-ref row (fx+ x (caar k)))])
;;                      (fl+ sum (fl* inflow (tail k)))))
;;                 0. filter))))
;;          (tail (iota (dec size))))
;;     trails*))

(define (box-blur line)
  (define kernel 3)
  (define radius (fx/ kernel 2))
  (define width (vector-length line))
  (let ([line* (make-vector width 0.)])
    (let slide ([r (sum (map (partial vector-ref line) (iota kernel)))]
                [x 0])
      (vector-set! line* (fx+ x radius) (fl/ r (inexact kernel)))
      (if (>= x (fx- width kernel)) line*
          (slide (fl+ r (vector-ref line (fx+ x kernel))
                    (fl- (vector-ref line x)))
                 (inc x))))))

(define (transpose M)
  (let* ([size (vector-length M)]
         [N (make-grid size)])
    (do ([y 0 (inc y)]) ((>= y size) N)
      (let ([row (vector-ref M y)])
        (do ([x 0 (inc x)]) ((>= x size))
          (vector-set! (vector-ref N x) y
                       (vector-ref row x)))))))

(define (diffuse trails)
  (let ([gauss (compose box-blur box-blur box-blur)])
    (transpose
     (list->vector
      (map gauss
           ((compose vector->list transpose list->vector)
            (map gauss (vector->list trails))))))))

(define (tick trails agents)
  (let ([agents (map (partial agent-update trails) agents)]
        [trails (diffuse (new-trails trails agents))])
    (values trails agents)))

(define (log-scale x) (if (< x 0.2) 0 (log (* 10 x) 2)))

(define (show-trails trails)
  (vector-for-each
   (lambda (row)
     (vector-for-each
      (λ x -> (let ([v (exact (round (log-scale x)))])
                (format #t "~s" (if (< 9 v) 'X v))))
      row)
     (print))
   trails))

(define (export-image trails)
  (define max-value
    (log-scale
     (apply max (vector->list
                 (vector-map (λ l -> (apply max (vector->list l))) trails)))))
  (define (rescale v)
    (exact (round (* (log-scale v) (/ 255 max-value)))))
  ((λ ppm ->
      (with-output-to-file "img.ppm" (λ -> (print ppm))
                           'replace))
   (apply string-append
          "P2\n"
          (number->string (vector-length trails)) " "
          (number->string (vector-length trails)) "\n"
          "255\n"
          (vector->list
           (vector-map
            (λ line ->
               (string-append
                (string-join
                 " "
                 (map (compose number->string rescale)
                      (vector->list line)))
                "\n"))
            trails)))))

(define (random-agent)
  (make-agent
   (make-vec2 (random 4.) (random 4.))
   (vec2-rotate (make-vec2 .8 0.) (random 6.2383))))

(export-image
 (time
  (let loop ([steps 240]
             [trails (make-grid 400)]
             [agents (map (λ _ -> (random-agent)) (iota 800))])
    (if (> steps 0)
        (call-with-values (λ -> (tick trails agents))
          (λ trails agents -> (loop (dec steps) trails agents)))
        trails))))

;; (let loop ([steps 10]
;;            [trails (make-grid 50)]
;;            [agents (map (λ _ -> (random-agent)) (iota 20))])
;;   (show-trails trails) (print)
;;   (when (> steps 0)
;;     (call-with-values (λ -> (tick trails agents))
;;       (λ trails agents -> (loop (dec steps) trails agents)))))

;; The model postulated by Jones employs both an agent-based layer (the data
;; map) and a continuum-based layer (the trail map). The data map consists of
;; many particles, while the trail map consists of a 2D grid of intensities
;; (similar to a pixel-based image). The data and trail map in turn affect each
;; other; the particles of the data map deposit material onto the trail map,
;; while those same particles sense values from the trail map in order to
;; determine aspects of their locomotion.

;; Each particle in the simulation has a heading angle, a location, and three
;; sensors (front left, front, front right). The sensor readings effect the
;; heading of the particle, causing it to rotate left or right (or stay facing
;; the same direction). The trail map undergoes a diffusion and decay process
;; every simulation step. A simple 3-by-3 mean filter is applied to simulate
;; diffusion of the particle trail, and then a multiplicative decay factor is
;; applied to simulate trail dissipation over time. The diagram below describes
;; the six sub-steps of a simulation tick.

;; Many of the parameters of this simulation are configurable, including sensor
;; distance, sensor size, sensor angle, step size, rotation angle, deposition
;; amount, decay factor, deposit size, diffuse size, decay factor, etc. For a
;; more detailed description check out the original paper.

;; There are several substantial differences between the model as described by
;; Jones and my implementation. In Jones’s original paper there is a collision
;; detection step that ensures that there is at most one particle in each grid
;; square. For my implementations I usually ignored this step, preferring the
;; patterns that arose without it. However, this step is crucial for exact
;; mimicry of the behavior of Physarum polycephalum, as it approximates a sort
;; of conservation of matter. Also (conveniently) this collision detection
;; removes any sort of sequential dependence, allowing for increased
;; computational parallelism.

