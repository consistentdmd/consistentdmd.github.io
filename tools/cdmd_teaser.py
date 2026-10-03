"""CDMD teaser animation (Manim Community Edition).

Particles leave the prior distribution and follow the student's velocity field to the
student distribution q0. Mid-path, the teacher's velocity field guides them
through the CDMD loss; as the loss shrinks the student flow transforms into
the teacher's, until q0 matches the teacher distribution p0.

Both flows are exact flow-matching velocities for Gaussian-mixture targets,
so the particle cloud follows a genuine probability path, and arrow opacity
tracks the marginal density at the current time.

Render (from the repo root):
    manim -qh --fps 60 tools/cdmd_teaser.py CDMDTeaser
"""

import numpy as np
from manim import *

config.background_color = "#111318"

STUDENT = "#4fc1a6"
TEACHER = "#f39a5c"
NOISE = "#9fb4c8"
LOSS = "#ff6b5f"
INK = "#e8ecef"
MUTED = "#8a96a3"

# ------------------------------------------------------------------ model
NOISE_C = np.array([-5.0, 0.35])
NOISE_S = 0.62


def gmm(means, sizes, angles, weights):
    covs = []
    for (sa, sb), a in zip(sizes, angles):
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        covs.append(R @ np.diag([sa ** 2, sb ** 2]) @ R.T)
    return dict(mu=np.array(means, float), cov=np.array(covs),
                w=np.array(weights, float), sizes=np.array(sizes, float),
                ang=np.array(angles, float))


P0 = gmm([[3.95, 2.25], [5.25, 1.05]], [[0.55, 0.24], [0.55, 0.24]],
         [np.deg2rad(40)] * 2, [0.5, 0.5])
Q0_INIT = gmm([[3.55, -1.55], [5.0, -2.25]], [[0.5, 0.3], [0.5, 0.3]],
              [np.deg2rad(-12), np.deg2rad(8)], [0.5, 0.5])


def student_gmm(alpha):
    lerp = lambda a, b: (1 - alpha) * a + alpha * b
    return gmm(lerp(Q0_INIT["mu"], P0["mu"]), lerp(Q0_INIT["sizes"], P0["sizes"]),
               lerp(Q0_INIT["ang"], P0["ang"]), lerp(Q0_INIT["w"], P0["w"]))


def _components(x, t, g):
    """Per-component marginal mean/cov of x_t and posterior pieces."""
    t = float(np.clip(t, 0.0, 0.995))
    m = (1 - t) * NOISE_C + t * g["mu"]                          # (K,2)
    A = t ** 2 * g["cov"] + ((1 - t) * NOISE_S) ** 2 * np.eye(2)  # (K,2,2)
    Ainv = np.linalg.inv(A)
    d = x[:, None, :] - m[None]                                    # (N,K,2)
    maha = np.einsum("nki,kij,nkj->nk", d, Ainv, d)
    logp = (np.log(g["w"])[None] - 0.5 * maha
            - 0.5 * np.log(np.linalg.det(A))[None] - np.log(2 * np.pi))
    return t, d, Ainv, logp


def velocity(x, t, g):
    """Exact flow-matching velocity E[x1 - x0 | x_t = x], x_t = (1-t)x0 + t x1."""
    t, d, Ainv, logp = _components(x, t, g)
    r = np.exp(logp - logp.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    post = g["mu"][None] + t * np.einsum("kij,kjl,nkl->nki", g["cov"], Ainv, d)
    ex1 = np.einsum("nk,nki->ni", r, post)
    return (ex1 - x) / (1 - t)


def density(x, t, g):
    _, _, _, logp = _components(x, t, g)
    return np.exp(logp).sum(1)


# particles and their trajectories on a grid of training progress values
N = 320
rng = np.random.default_rng(4)
Z = NOISE_C + NOISE_S * rng.standard_normal((N, 2))
TAUS = np.linspace(0, 1, 151)
ALPHAS = np.linspace(0, 1, 26)


def integrate(g):
    out = np.empty((TAUS.size, N, 2))
    x = Z.copy()
    out[0] = x
    for i in range(TAUS.size - 1):
        t0, t1 = TAUS[i], min(TAUS[i + 1], 0.995)
        h = t1 - t0
        k1 = velocity(x, t0, g)
        k2 = velocity(x + h / 2 * k1, t0 + h / 2, g)
        k3 = velocity(x + h / 2 * k2, t0 + h / 2, g)
        k4 = velocity(x + h * k3, t1, g)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = x
    return out


TRAJ = np.stack([integrate(student_gmm(a)) for a in ALPHAS])  # (A,T,N,2)

# The flows above are computed in a latent plane and drawn through a smooth
# vertical arc y -> y + b * bump(x). The teacher arcs up, the untrained
# student sags down, and the student's arc follows its training progress.
# The map has unit Jacobian determinant, so densities carry over unchanged.
ARC_TEACHER, ARC_STUDENT, ARC_W = 1.15, -1.5, 2.7


def arc_amount(alpha):
    return (1 - alpha) * ARC_STUDENT + alpha * ARC_TEACHER


def bump(x):
    return np.exp(-(x / ARC_W) ** 2)


def to_display(xy, b):
    xy = np.array(xy, float)
    out = xy.copy()
    out[..., 1] += b * bump(xy[..., 0])
    return out


def to_latent(xy, b):
    out = np.array(xy, float).copy()
    out[..., 1] -= b * bump(out[..., 0])
    return out


def push_velocity(xy_latent, v, b):
    dbump = -2 * xy_latent[:, 0] / ARC_W ** 2 * bump(xy_latent[:, 0])
    out = v.copy()
    out[:, 1] += b * dbump * v[:, 0]
    return out


def display_velocity(xy, t, g, b):
    lat = to_latent(xy, b)
    return push_velocity(lat, velocity(lat, t, g), b)


def display_density(xy, t, g, b):
    return density(to_latent(xy, b), t, g)


def interp(arr, grid, v):
    v = float(np.clip(v, grid[0], grid[-1]))
    i = min(int(np.searchsorted(grid, v, side="right")) - 1, grid.size - 2)
    f = (v - grid[i]) / (grid[i + 1] - grid[i])
    return (1 - f) * arr[i] + f * arr[i + 1]


def positions(alpha, tau):
    return to_display(interp(interp(TRAJ, ALPHAS, alpha), TAUS, tau), arc_amount(alpha))


def path_upto(alpha, tau, idx):
    tr = to_display(interp(TRAJ, ALPHAS, alpha)[:, idx], arc_amount(alpha))
    n = int(np.clip(tau, 0, 1) * (TAUS.size - 1))
    pts = list(tr[: n + 1]) + [interp(tr, TAUS, tau)]
    return np.array(pts)


def p3(xy):
    return np.array([xy[0], xy[1], 0.0])


# particle followed through the guidance step: near the middle of the cloud
_mid = positions(0, 0.5)
FOCUS = int(np.argmin(np.linalg.norm(_mid - np.median(_mid, 0) - [0.25, 0.1], axis=1)))
TRACE = rng.choice(N, 14, replace=False)

GX, GY = np.meshgrid(np.arange(-6.6, 6.7, 0.52), np.arange(-2.75, 3.35, 0.52))
GRID = np.stack([GX.ravel(), GY.ravel()], 1)


# ------------------------------------------------------------------ helpers
def field_arrows(g, b, tau, color, max_len=0.44, base=0.13, width=2.3):
    t = min(tau, 0.93)
    v = display_velocity(GRID, t, g, b)
    p = display_density(GRID, t, g, b)
    p = (p / p.max()) ** 0.28
    sp = np.linalg.norm(v, axis=1) + 1e-9
    L = max_len * np.tanh(sp / 6.0)
    arrows = VGroup()
    for (x, y), d, l, op in zip(GRID, v / sp[:, None], L, p):
        a = base + (1 - base) * op
        start = np.array([x, y, 0]) - 0.5 * l * np.append(d, 0)
        end = np.array([x, y, 0]) + 0.5 * l * np.append(d, 0)
        arrows.add(Arrow(start, end, buff=0, stroke_width=width,
                         max_tip_length_to_length_ratio=0.38,
                         max_stroke_width_to_length_ratio=12,
                         color=color).set_opacity(a))
    return arrows


def blob(g, b, color, opacity=0.1, levels=(2.6, 2.1, 1.6, 1.15, 0.7)):
    grp = VGroup()
    for mu, (sa, sb), ang in zip(g["mu"], g["sizes"], g["ang"]):
        for r in levels:
            e = Ellipse(width=2 * sa * r, height=2 * sb * r)
            e.set_fill(color, opacity=opacity).set_stroke(width=0)
            grp.add(e.rotate(ang).move_to(p3(to_display(mu, b))))
    return grp


def outline(g, b, color, r=2.2, width=2.6):
    grp = VGroup()
    for mu, (sa, sb), ang in zip(g["mu"], g["sizes"], g["ang"]):
        e = DashedVMobject(Ellipse(width=2 * sa * r, height=2 * sb * r,
                                   stroke_color=color, stroke_width=width),
                           num_dashes=30)
        grp.add(e.rotate(ang).move_to(p3(to_display(mu, b))))
    return grp


def noise_blob():
    grp = VGroup()
    for r in (2.6, 2.1, 1.6, 1.1, 0.6):
        grp.add(Circle(radius=NOISE_S * r).set_fill(NOISE, 0.07)
                .set_stroke(width=0).move_to(p3(NOISE_C)))
    return grp


def swap(old, new):
    return [FadeOut(old, shift=UP * 0.15), FadeIn(new, shift=UP * 0.15)]


def caption(*parts, colors=None):
    tex = Tex(*parts, font_size=38, color=INK)
    for i, c in (colors or {}).items():
        tex[i].set_color(c)
    return tex.to_edge(DOWN, buff=0.32)


# ------------------------------------------------------------------ scene
class CDMDTeaser(Scene):
    def construct(self):
        tau = ValueTracker(0.0)
        alpha = ValueTracker(0.0)

        # --- 1. prior distribution and the teacher's target ------------------
        noise = noise_blob()
        p1_lab = MathTex("p_1 = q_1", font_size=44, color=NOISE).next_to(noise, DOWN, buff=0.05)
        p0 = blob(P0, ARC_TEACHER, TEACHER, opacity=0.12)
        p0_lab = VGroup(MathTex("p_0", font_size=44, color=TEACHER),
                        Tex("teacher", font_size=26, color=TEACHER)).arrange(DOWN, buff=0.06)
        p0_lab.move_to(p3(to_display(P0["mu"][1], ARC_TEACHER)) + [1.05, 1.3, 0])

        dots = VGroup(*[Dot(p3(z), radius=0.034, color=NOISE) for z in Z])

        def move_dots(m):
            pos = positions(alpha.get_value(), tau.get_value())
            col = interpolate_color(ManimColor(NOISE), ManimColor(STUDENT),
                                    smooth(min(1, tau.get_value() * 1.6)))
            for d, xy in zip(m, pos):
                d.move_to(p3(xy))
                d.set_color(col)

        dots.add_updater(move_dots)

        cap = caption(r"Start from the prior distribution ", r"$p_1 = q_1$", colors={1: NOISE})
        self.play(FadeIn(noise), FadeIn(p1_lab), FadeIn(cap, shift=UP * 0.2),
                  LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.004),
                  run_time=1.6)
        self.play(FadeIn(p0, scale=0.9), FadeIn(p0_lab), run_time=0.9)
        self.wait(0.4)

        # --- 2. student flow along its velocity field ---------------------
        s_field = always_redraw(lambda: field_arrows(
            student_gmm(alpha.get_value()), arc_amount(alpha.get_value()),
            tau.get_value(), STUDENT))
        traces = always_redraw(lambda: VGroup(*[
            VMobject(stroke_color=STUDENT, stroke_width=1.6, stroke_opacity=0.55)
            .set_points_smoothly([p3(q) for q in path_upto(alpha.get_value(), tau.get_value(), i)])
            for i in TRACE]) if tau.get_value() > 0.02 else VGroup())

        cap2 = caption(r"Particles follow the ", r"student", r" velocity field",
                       colors={1: STUDENT})
        self.add(traces)
        self.bring_to_front(dots)
        self.play(FadeIn(s_field), *swap(cap, cap2), run_time=1.0)
        self.bring_to_front(dots)
        self.play(tau.animate.set_value(1.0), run_time=4.2, rate_func=smooth)

        def q0_shape():
            a = alpha.get_value()
            g, b = student_gmm(a), arc_amount(a)
            return VGroup(blob(g, b, STUDENT, opacity=0.09 * (1 - smooth(a))),
                          outline(g, b, STUDENT))

        q0 = always_redraw(q0_shape)
        q0_lab = always_redraw(lambda: VGroup(
            MathTex("q_0", font_size=44, color=STUDENT),
            Tex("student", font_size=26, color=STUDENT)).arrange(DOWN, buff=0.06)
            .move_to(p3(to_display(student_gmm(alpha.get_value())["mu"][1],
                                   arc_amount(alpha.get_value()))) + [1.3, -0.75, 0]))
        cap3 = caption(r"\dots landing on the student distribution ", r"$q_0$", r", not the teacher's ", r"$p_0$",
                       colors={1: STUDENT, 3: TEACHER})
        self.play(FadeIn(q0), FadeIn(q0_lab), *swap(cap2, cap3), run_time=1.0)
        self.bring_to_front(dots)
        self.wait(1.0)

        # --- 3. teacher guidance at an intermediate state -----------------
        self.play(FadeOut(traces), run_time=0.4)
        self.remove(traces)
        self.play(tau.animate.set_value(0.0), run_time=1.0, rate_func=smooth)
        self.play(tau.animate.set_value(0.5), run_time=1.4, rate_func=smooth)

        t_field = field_arrows(P0, ARC_TEACHER, 0.5, TEACHER, width=2.6)
        cap4 = caption(r"The ", r"teacher", r" velocity guides each particle through ",
                       r"$\mathcal{L}_{\mathrm{CDMD}}$", colors={1: TEACHER, 3: LOSS})
        self.play(FadeIn(t_field), *swap(cap3, cap4), run_time=1.0)
        self.bring_to_front(s_field, dots)

        def focus_xy():
            return positions(alpha.get_value(), tau.get_value())[FOCUS]

        def vecs():
            a = alpha.get_value()
            x = focus_xy()[None]
            vs = display_velocity(x, 0.5, student_gmm(a), arc_amount(a))[0]
            vt = display_velocity(x, 0.5, P0, ARC_TEACHER)[0]
            return x[0], vs, vt

        SCALE = 0.17

        def guide():
            x, vs, vt = vecs()
            o = p3(x)
            ts, tt = o + SCALE * p3(vs), o + SCALE * p3(vt)
            g = VGroup(
                Arrow(o, tt, buff=0, color=TEACHER, stroke_width=6,
                      max_tip_length_to_length_ratio=0.16),
                Arrow(o, ts, buff=0, color=STUDENT, stroke_width=6,
                      max_tip_length_to_length_ratio=0.16),
            )
            gap = np.linalg.norm(tt - ts)
            if gap > 0.04:
                g.add(DashedLine(ts, tt, color=LOSS, stroke_width=4,
                                 dash_length=0.08).set_opacity(min(1, gap / 0.3)))
            g.add(Dot(o, radius=0.085, color=WHITE))
            return g

        guide_m = always_redraw(guide)

        def guide_labels():
            x, vs, vt = vecs()
            o = p3(x)
            ts, tt = o + SCALE * p3(vs), o + SCALE * p3(vt)
            labs = VGroup(
                MathTex(r"v_{\text{teacher}}", font_size=34, color=TEACHER).next_to(tt, UP, buff=0.12),
                MathTex(r"x_t \sim q_t", font_size=34, color=WHITE).next_to(o, UL, buff=0.12),
            )
            if alpha.get_value() < 0.9:
                labs.add(MathTex(r"v_{\text{student}}", font_size=34, color=STUDENT)
                         .next_to(ts, DOWN, buff=0.12)
                         .set_opacity(1 - smooth(alpha.get_value() / 0.9)))
            gap = np.linalg.norm(tt - ts)
            if gap > 0.15:
                labs.add(MathTex(r"\mathcal{L}_{\mathrm{CDMD}}", font_size=34, color=LOSS)
                         .move_to((ts + tt) / 2 + RIGHT * 0.75)
                         .set_opacity(min(1, gap / 0.5)))
            for m in labs:
                m.add_background_rectangle(color=config.background_color, opacity=0.75, buff=0.05)
            return labs

        labels_m = always_redraw(guide_labels)
        self.play(FadeIn(guide_m), FadeIn(labels_m), run_time=0.9)
        self.wait(0.8)

        # loss meter, top-left
        def residual(a):
            x = positions(a, 0.5)
            d = (display_velocity(x, 0.5, student_gmm(a), arc_amount(a))
                 - display_velocity(x, 0.5, P0, ARC_TEACHER))
            return float(np.mean(np.sum(d ** 2, 1)))

        R0 = residual(0.0)
        meter_lab = MathTex(r"\mathcal{L}_{\mathrm{CDMD}}", font_size=36, color=LOSS)
        meter_lab.to_corner(UL, buff=0.45)
        track = RoundedRectangle(width=2.6, height=0.16, corner_radius=0.08,
                                 stroke_width=0, fill_color=WHITE, fill_opacity=0.12)
        track.next_to(meter_lab, RIGHT, buff=0.3)
        bar = always_redraw(lambda: RoundedRectangle(
            width=max(0.02, 2.6 * residual(alpha.get_value()) / R0), height=0.16,
            corner_radius=0.08, stroke_width=0, fill_color=LOSS, fill_opacity=1)
            .align_to(track, LEFT).align_to(track, UP))
        meter = VGroup(meter_lab, track)
        for m in meter:
            m.add_background_rectangle(color=config.background_color, opacity=0.8, buff=0.08)

        cap5 = caption(r"Minimizing ", r"$\mathcal{L}_{\mathrm{CDMD}}$",
                       r" transforms the student flow into the teacher's",
                       colors={1: LOSS})
        self.play(FadeIn(meter), FadeIn(bar), *swap(cap4, cap5), run_time=0.8)
        self.play(alpha.animate.set_value(1.0), run_time=5.0, rate_func=smooth)
        self.wait(0.4)

        # --- 4. continue the flow: q0 lands on p0 -------------------------
        self.play(FadeOut(guide_m), FadeOut(labels_m), FadeOut(t_field), run_time=0.6)
        self.remove(guide_m, labels_m)
        traces2 = always_redraw(lambda: VGroup(*[
            VMobject(stroke_color=STUDENT, stroke_width=1.6, stroke_opacity=0.55)
            .set_points_smoothly([p3(q) for q in path_upto(1.0, tau.get_value(), i)])
            for i in TRACE]))
        self.add(traces2)
        self.bring_to_front(dots)
        self.play(tau.animate.set_value(1.0), run_time=2.4, rate_func=smooth)

        cap6 = caption(r"Eventually ", r"$q_0 \approx p_0$", colors={1: STUDENT})
        q0_final = VGroup(MathTex(r"q_0 \approx p_0", font_size=44, color=STUDENT))
        q0_final.move_to(p3(to_display(P0["mu"][1], ARC_TEACHER)) + [0.1, -1.25, 0])
        self.play(FadeOut(q0_lab), FadeIn(q0_final), FadeOut(p0_lab),
                  *swap(cap5, cap6), run_time=1.2)
        self.wait(1.4)

        # --- 5. one-step generation -----------------------------------------
        self.play(FadeOut(traces2), FadeOut(s_field), run_time=0.6)
        self.remove(traces2, s_field)
        dots.clear_updaters()
        final = positions(1.0, 1.0)
        jumps = VGroup(*[Line(p3(Z[i]), p3(final[i]), stroke_width=1.4,
                              color=STUDENT, stroke_opacity=0.35) for i in TRACE])
        cap7 = caption(r"\dots and the student gets there in ", r"one step",
                       colors={1: STUDENT})
        self.play(*[d.animate.move_to(p3(z)).set_color(NOISE) for d, z in zip(dots, Z)],
                  *swap(cap6, cap7), run_time=1.0)
        self.play(Create(jumps), *[d.animate.move_to(p3(xy)).set_color(STUDENT)
                                   for d, xy in zip(dots, final)],
                  run_time=0.9, rate_func=rush_into)
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
