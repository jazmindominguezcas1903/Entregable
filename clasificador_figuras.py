"""
Clasificador de figuras geométricas con OpenCV (heurística + búsqueda de hipótesis)
Alumna: Domínguez Chávez Joselyn Jazmín | Matrícula: 24305015
Profesor: Javier García Lara

Reconoce 5 figuras (circulo, cuadrado, triangulo, pentagono, estrella) + NULL.
Uso:   python clasificador_figuras.py ruta/imagen.png
       python clasificador_figuras.py            (abre un selector de archivos)
Requiere: pip install opencv-python numpy
"""
import math, sys
import cv2
import numpy as np

AUTOR = "Dominguez Chavez Joselyn Jazmin | Mat. 24305015 | Prof. Javier Garcia Lara"
UMBRAL_NULL = 0.35   # si la mejor hipótesis puntúa menos -> NULL

# ---------- Base de conocimiento: prototipo ideal de cada figura ----------
# n = nº de vértices, c = circularidad (4πA/P²), s = solidez (A/A_hull), a = aspecto
PROTOS = {
    "circulo":   dict(n=10, c=0.90,  s=0.99, a=1.0, many=True),
    "triangulo": dict(n=3,  c=0.56,  s=0.99, a=None),
    "cuadrado":  dict(n=4,  c=0.79,  s=0.99, a=1.0),
    "pentagono": dict(n=5,  c=0.79,  s=0.99, a=None),
    "estrella":  dict(n=10, c=0.27,  s=0.52, a=None, nt=2.0),
}


def preprocesar(img):
    """Gris -> desenfoque -> umbral de Otsu -> la figura queda en blanco."""
    gris = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    borde = np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]])
    if borde.mean() > 127:                      # decisión: fondo claro -> invertir
        th = cv2.bitwise_not(th)
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def extraer_rasgos(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    aprox = cv2.approxPolyDP(cnt, 0.02 * per, True)
    hull = cv2.convexHull(cnt)
    (w, h) = cv2.minAreaRect(cnt)[1]
    rasgos = dict(
        n=len(aprox),
        c=4 * math.pi * area / (per ** 2),
        s=area / max(cv2.contourArea(hull), 1e-6),
        a=min(w, h) / max(w, h, 1e-6),
        area=area,
    )
    return rasgos, aprox, hull


def puntuar(f, p):
    """Qué tan bien encaja el rasgo f en el prototipo p (0..1, gaussiana)."""
    if p.get("many"):
        dn = max(0, 7 - f["n"]) / 1.0           # círculo: muchos vértices
    else:
        dn = (f["n"] - p["n"]) / p.get("nt", 0.5)
    tc = 0.06 if p.get("many") else 0.10
    d = dn ** 2 + ((f["c"] - p["c"]) / tc) ** 2 + ((f["s"] - p["s"]) / 0.15) ** 2
    if p["a"] is not None:
        d += ((f["a"] - p["a"]) / 0.15) ** 2
    return math.exp(-d / 2)


def clasificar(f, log):
    """Búsqueda heurística: evalúa todas las hipótesis y elige la mejor."""
    puntajes = {k: puntuar(f, p) for k, p in PROTOS.items()}
    orden = sorted(puntajes.items(), key=lambda kv: kv[1], reverse=True)
    (mejor, s1), (rival, s2) = orden[0], orden[1]
    if f["n"] == 3 and f["s"] > 0.9:
        log.append("Regla directa: 3 vértices + convexo -> triangulo")
    if s1 < UMBRAL_NULL:                        # decisión: ninguna hipótesis convence
        log.append(f"Mejor puntaje {s1:.2f} < {UMBRAL_NULL} -> NULL")
        return "NULL", s1 * 100, puntajes
    conf = s1 * (s1 / (s1 + s2 + 1e-9)) * 100   # penaliza si hay rival cercano
    log.append(f"Mejor hipótesis: {mejor} ({s1:.2f}) vs 2da {rival} ({s2:.2f})")
    return mejor, conf, puntajes


def grafo_figura(aprox):
    """Grafo ciclo: nodo = vértice, arista = lado, peso = longitud (px).
    Nodo de inicio = vértice más alto (y mínima, luego x mínima)."""
    pts = aprox[:, 0, :].astype(float)
    ini = min(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))
    pts = np.roll(pts, -ini, axis=0)            # el nodo 0 es el de inicio
    n = len(pts)
    aristas = [(i, (i + 1) % n, float(np.linalg.norm(pts[(i + 1) % n] - pts[i])))
               for i in range(n)]
    return pts, aristas


def analizar(img):
    """Pipeline completo. Devuelve dict con todo (o None si no hay contorno)."""
    log = []
    th = preprocesar(img)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.002 * th.size]
    if not cnts:
        log.append("Sin contornos -> NULL")
        return dict(nombre="NULL", conf=100.0, th=th, log=log, cnt=None)
    cnt = max(cnts, key=cv2.contourArea)
    f, aprox, hull = extraer_rasgos(cnt)
    log.insert(0, f"Rasgos: n={f['n']} circ={f['c']:.3f} sol={f['s']:.3f} asp={f['a']:.3f}")
    nombre, conf, punt = clasificar(f, log)
    return dict(nombre=nombre, conf=conf, f=f, aprox=aprox, hull=hull, cnt=cnt,
                th=th, puntajes=punt, log=log)


# ----------------------------- dibujo -----------------------------
def lienzo(img, tam=420):
    h, w = img.shape[:2]
    k = tam / max(h, w)
    r = cv2.resize(img, (int(w * k), int(h * k)))
    out = np.full((tam, tam, 3), 255, np.uint8)
    y, x = (tam - r.shape[0]) // 2, (tam - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def dibujar_rasgos(R):
    vis = cv2.cvtColor(R["th"], cv2.COLOR_GRAY2BGR)
    if R["cnt"] is None:
        return vis
    cv2.drawContours(vis, [R["cnt"]], -1, (0, 200, 255), 2)
    cv2.drawContours(vis, [R["hull"]], -1, (0, 160, 0), 1)
    box = cv2.boxPoints(cv2.minAreaRect(R["cnt"])).astype(int)
    cv2.polylines(vis, [box], True, (255, 0, 0), 1)
    for p in R["aprox"]:
        cv2.circle(vis, tuple(int(v) for v in p[0]), 5, (0, 0, 255), -1)
    return vis


def dibujar_grafo(R, tam=420):
    """Dibujo del grafo: nodos numerados, INICIO en el nodo 0, peso en cada arista."""
    g = np.full((tam, tam, 3), 255, np.uint8)
    if R["cnt"] is None:
        cv2.putText(g, "NULL: sin grafo", (110, 210), 0, 0.8, (0, 0, 0), 2)
        return g
    pts, ar = grafo_figura(R["aprox"])
    mn, mx = pts.min(0), pts.max(0)
    esc = (tam - 120) / max((mx - mn).max(), 1)
    P = (pts - mn) * esc + 60
    for i, j, w in ar:
        a, b = P[i].astype(int), P[j].astype(int)
        cv2.line(g, tuple(a), tuple(b), (110, 110, 110), 2)
        m = (a + b) // 2
        cv2.putText(g, f"{w:.0f}", (int(m[0]) + 4, int(m[1]) - 4), 0, 0.5, (160, 0, 0), 1)
    for i, p in enumerate(P.astype(int)):
        col = (0, 140, 0) if i == 0 else (0, 0, 220)
        cv2.circle(g, tuple(p), 12, col, -1)
        cv2.putText(g, f"v{i}", (p[0] - 9, p[1] + 4), 0, 0.4, (255, 255, 255), 1)
    p0 = P[0].astype(int)
    cv2.putText(g, "INICIO", (p0[0] - 25, p0[1] - 20), 0, 0.5, (0, 120, 0), 2)
    return g


def panel_final(img, R):
    paneles = [lienzo(img), lienzo(dibujar_rasgos(R)), dibujar_grafo(R)]
    for p, t in zip(paneles, ["1. Original", "2. Rasgos detectados",
                              "3. Grafo (peso = longitud px)"]):
        cv2.putText(p, t, (8, 20), 0, 0.55, (0, 0, 0), 1)
    cuerpo = np.hstack(paneles)
    banner = np.full((70, cuerpo.shape[1], 3), 30, np.uint8)
    cv2.putText(banner, f"Figura: {R['nombre']}   Confianza: {R['conf']:.1f}%",
                (15, 35), 0, 0.9, (255, 255, 255), 2)
    cv2.putText(banner, AUTOR, (15, 60), 0, 0.5, (200, 200, 200), 1)
    return np.vstack([banner, cuerpo])


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else None
    if not ruta:
        import tkinter as tk
        from tkinter import filedialog
        tk.Tk().withdraw()
        ruta = filedialog.askopenfilename(title="Elige una imagen")
    img = cv2.imread(ruta)
    if img is None:
        sys.exit("No se pudo leer la imagen.")
    R = analizar(img)
    print("\n".join(R["log"]))
    if R.get("puntajes"):
        for k, v in sorted(R["puntajes"].items(), key=lambda kv: -kv[1]):
            print(f"  {k:<10}{v * 100:6.1f}%")
    if R["cnt"] is not None:
        _, ar = grafo_figura(R["aprox"])
        print("Nodo de inicio: v0 | Aristas:", ", ".join(f"v{i}-v{j}={w:.0f}px" for i, j, w in ar))
    print(f"==> FIGURA: {R['nombre']}  |  CONFIANZA: {R['conf']:.1f}%")
    final = panel_final(img, R)
    cv2.imwrite("resultado.png", final)
    cv2.imshow("Clasificador de figuras (cualquier tecla para salir)", final)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
