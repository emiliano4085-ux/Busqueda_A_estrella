"""
Simulador gráfico de búsqueda en un mapa de calles: Dijkstra, A* y A* ponderada.

El mapa es un GRAFO:
    Vértices = cruces (nombrados como en una guía de calles: A1, B3, ...)
    Aristas  = calles, cada una con una longitud real (km) y un tipo:
        Avenida   60 km/h
        Calle     30 km/h
        Tráfico   12 km/h
        Cerrada   (en obras, no se puede pasar)
    Costo de una arista = tiempo de viaje en minutos = longitud / velocidad.

Hay un río que solo se cruza por puentes y una avenida diagonal que atraviesa
las manzanas.

    Dijkstra        f(n) = g(n)
    A*              f(n) = g(n) + h(n)
    A* ponderada    f(n) = g(n) + w · h(n)

h(n) = distancia a la meta / velocidad máxima (60 km/h), en minutos.
    Euclidiana (línea recta): admisible -> A* encuentra el óptimo.
    Manhattan: puede SOBREESTIMAR por la diagonal y las calles chuecas
               -> A* puede no ser óptimo (comparar con Dijkstra).
    Chebyshev: admisible pero menos informada (expande más).

Requiere Python 3.8+ con tkinter (incluido en Windows y macOS;
en Linux: sudo apt install python3-tk).

Ejecutar:  python busqueda_informada_gui.py
"""

import heapq
import math
import random
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Paleta (la del primer simulador)
# ---------------------------------------------------------------------------

C = {
    "bg": "#070b14",
    "panel": "#111827",
    "panel2": "#090e1a",
    "border": "#1f293d",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "dim": "#475569",
    "accent": "#38bdf8",

    "start": "#22c55e",
    "goal": "#ef4444",
    "frontier": "#eab308",
    "visited": "#4f46e5",
    "tree": "#818cf8",
    "current": "#ec4899",
    "testing": "#f97316",
    "path": "#10b981",

    "ground": "#0b1120",
    "river": "#0b2a4a",
    "river_edge": "#1e4b7a",
    "node": "#1e293b",

    "flow_box": "#090e1a",
    "flow_line": "#334155",
    "flow_active": "#0c2d48",
    "flow_ok": "#064e3b",
    "flow_bad": "#450a0a",

    "btn_primary": "#0284c7",
    "btn_step": "#6366f1",
    "pill_on": "#0c2d48",
}

# tipo: (nombre, velocidad km/h, grosor, color)
CALLES = {
    "avenida": ("Avenida", 60, 8, "#3b4a63"),
    "calle": ("Calle", 30, 4, "#2a3850"),
    "trafico": ("Tráfico", 12, 4, "#9f1239"),
    "cerrada": ("Cerrada", None, 2, "#475569"),
}
CICLO_TIPO = {"avenida": "calle", "calle": "trafico", "trafico": "cerrada", "cerrada": "avenida"}
V_MAX = 60  # km/h, para que h sea admisible

FONT = "Segoe UI"
MONO = "Consolas"

ALGORITMOS = [("DIJ", "Dijkstra"), ("AST", "A*"), ("WAS", "A* ponderada")]
NOMBRE_ALG = dict(ALGORITMOS)
FORMULA = {"DIJ": "f = g(n)", "AST": "f = g(n) + h(n)", "WAS": "f = g(n) + w·h(n)"}
HEURISTICAS = [("L2", "Euclidiana"), ("L1", "Manhattan"), ("LI", "Chebyshev")]
NOMBRE_HEUR = dict(HEURISTICAS)
TAMANOS = [("6x5", "Barrio 6×5"), ("8x6", "Colonia 8×6"), ("10x7", "Ciudad 10×7")]


def num(v, dec=2):
    if v is None:
        return "—"
    if v == math.inf:
        return "∞"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{dec}f}"


# ---------------------------------------------------------------------------
# Grafo de calles
# ---------------------------------------------------------------------------

class Grafo:
    def __init__(self, nodos, aristas, rio=None, etiquetas=None, origen=None, destino=None):
        self.nodos = dict(nodos)            # id -> (x_km, y_km)
        self.aristas = dict(aristas)        # frozenset({a, b}) -> tipo
        self.rio = rio                      # (y_centro, semiancho, x_min, x_max, fase)
        self.etiquetas = list(etiquetas or [])
        self.orden = sorted(self.nodos, key=lambda n: (n[0], int(n[1:])))
        self.origen = origen or self.orden[0]
        self.destino = destino or self.orden[-1]
        self._ady()

    def _ady(self):
        self.ady = {n: [] for n in self.nodos}
        for par, tipo in self.aristas.items():
            a, b = tuple(par)
            self.ady[a].append(b)
            self.ady[b].append(a)
        for n in self.ady:
            self.ady[n].sort(key=lambda m: self.orden.index(m))

    def tipo(self, a, b):
        return self.aristas[frozenset((a, b))]

    def longitud(self, a, b):
        (x1, y1), (x2, y2) = self.nodos[a], self.nodos[b]
        return math.hypot(x1 - x2, y1 - y2)

    def costo(self, a, b):
        v = CALLES[self.tipo(a, b)][1]
        if v is None:
            return math.inf
        return self.longitud(a, b) / v * 60  # minutos

    def copia(self):
        return Grafo(self.nodos, self.aristas, self.rio, self.etiquetas, self.origen, self.destino)

    def alcanzable(self, a, b):
        vistos, pila = {a}, [a]
        while pila:
            n = pila.pop()
            if n == b:
                return True
            for m in self.ady[n]:
                if m not in vistos and self.tipo(n, m) != "cerrada":
                    vistos.add(m)
                    pila.append(m)
        return False


def _conexo(nodos, aristas):
    ady = {n: [] for n in nodos}
    for par in aristas:
        a, b = tuple(par)
        ady[a].append(b)
        ady[b].append(a)
    inicio = next(iter(nodos))
    vistos, pila = {inicio}, [inicio]
    while pila:
        for m in ady[pila.pop()]:
            if m not in vistos:
                vistos.add(m)
                pila.append(m)
    return len(vistos) == len(nodos)


def mapa_ciudad(cols=8, filas=6):
    """
    Genera una colonia: cuadrícula irregular de cruces, una avenida horizontal,
    una vertical, una diagonal, un río con 2 o 3 puentes, calles con tráfico y
    algunas manzanas fusionadas (calles eliminadas).
    """
    sep = 0.4     # km entre cruces
    jit = 0.07
    letra = lambda i: "ABCDEFGHIJKL"[i]
    nid = lambda i, j: f"{letra(i)}{j + 1}"

    nodos = {}
    for i in range(cols):
        for j in range(filas):
            nodos[nid(i, j)] = (i * sep + random.uniform(-jit, jit), j * sep + random.uniform(-jit, jit))

    aristas = {}
    for i in range(cols):
        for j in range(filas):
            if i + 1 < cols:
                aristas[frozenset((nid(i, j), nid(i + 1, j)))] = "calle"
            if j + 1 < filas:
                aristas[frozenset((nid(i, j), nid(i, j + 1)))] = "calle"

    protegidas = set()
    etiquetas = []

    # Río entre la fila r y r+1, con puentes
    r = random.randint(1, filas - 3) if filas >= 4 else 0
    ys = [nodos[nid(i, r)][1] for i in range(cols)] + [nodos[nid(i, r + 1)][1] for i in range(cols)]
    y_rio = sum(ys) / len(ys)
    puentes = set(random.sample(range(cols), k=min(cols, 2 if cols <= 6 else 3)))
    for i in range(cols):
        par = frozenset((nid(i, r), nid(i, r + 1)))
        if i in puentes:
            protegidas.add(par)
        else:
            del aristas[par]
    rio = (y_rio, sep * 0.14, -sep * 0.6, (cols - 1) * sep + sep * 0.6, random.uniform(0, 6.28))

    # Avenida horizontal (lejos del río)
    filas_ok = [j for j in range(filas) if j not in (r, r + 1)] or [0]
    fh = random.choice(filas_ok)
    for i in range(cols - 1):
        par = frozenset((nid(i, fh), nid(i + 1, fh)))
        aristas[par] = "avenida"
        protegidas.add(par)
    etiquetas.append(("Av. Central", nid(0, fh), nid(1, fh)))

    # Avenida vertical (cruza el río por su puente)
    cv_ = random.choice(sorted(puentes))
    for j in range(filas - 1):
        par = frozenset((nid(cv_, j), nid(cv_, j + 1)))
        aristas[par] = "avenida"
        protegidas.add(par)
    etiquetas.append(("Av. Norte-Sur", nid(cv_, 0), nid(cv_, 1)))

    # Avenida diagonal que atraviesa manzanas (aquí Manhattan deja de ser admisible)
    largo = min(cols, filas) - 2
    i0 = random.randint(0, max(0, cols - largo - 1))
    j0 = random.randint(0, max(0, filas - largo - 1))
    sube = random.random() < 0.5
    for k in range(largo):
        if sube:
            a, b = nid(i0 + k, filas - 1 - (j0 + k)), nid(i0 + k + 1, filas - 2 - (j0 + k))
        else:
            a, b = nid(i0 + k, j0 + k), nid(i0 + k + 1, j0 + k + 1)
        par = frozenset((a, b))
        aristas[par] = "avenida"
        protegidas.add(par)
        if k == 0:
            etiquetas.append(("Diagonal", a, b))

    # Manzanas fusionadas: quitar calles sin desconectar
    candidatas = [p for p in aristas if p not in protegidas]
    random.shuffle(candidatas)
    for par in candidatas[: int(len(candidatas) * 0.16)]:
        tipo = aristas.pop(par)
        if not _conexo(nodos, aristas):
            aristas[par] = tipo

    # Tráfico
    for par in aristas:
        if par not in protegidas and random.random() < 0.2:
            aristas[par] = "trafico"

    # Origen al oeste, destino al este
    origen = nid(random.randint(0, 1), random.randint(0, filas - 1))
    destino = nid(cols - 1 - random.randint(0, 1), random.randint(0, filas - 1))
    return Grafo(nodos, aristas, rio, etiquetas, origen, destino)


# ---------------------------------------------------------------------------
# Algoritmo: primero el mejor con lista ABIERTA (cola de prioridad)
# ---------------------------------------------------------------------------

class Buscador:
    """
    Una sola implementación para los tres algoritmos; solo cambia el peso de h:
        Dijkstra -> 0,   A* -> 1,   A* ponderada -> w.
    Es un generador: cada `yield` es una micro-decisión del diagrama de flujo.
    """

    def __init__(self, grafo, algoritmo="AST", w=2.0, heuristica="L2"):
        self.G = grafo
        self.algoritmo = algoritmo
        self.heur = heuristica
        self.peso = {"DIJ": 0.0, "AST": 1.0, "WAS": float(w)}[algoritmo]
        self.abierta = []        # heap de (f, desempate, contador, nodo, g)
        self.cerrados = set()
        self.g = {}
        self.padre = {}
        self.cont = 0
        self.expandidos = 0
        self.generados = 0
        self.obsoletos = 0
        self.max_abierta = 0
        self.ruta = []

    def distancia(self, n, tipo=None):
        (x1, y1), (x2, y2) = self.G.nodos[n], self.G.nodos[self.G.destino]
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        tipo = tipo or self.heur
        if tipo == "L1":
            return dx + dy
        if tipo == "L2":
            return math.hypot(dx, dy)
        return max(dx, dy)

    def h(self, n):
        return self.distancia(n) / V_MAX * 60  # minutos

    def f(self, n, g):
        return g + self.peso * self.h(n)

    def _push(self, n, g):
        hv = self.h(n)
        fv = g + self.peso * hv
        self.cont += 1
        heapq.heappush(self.abierta, (fv, hv if self.peso > 0 else 0, self.cont, n, g))
        self.generados += 1
        self.max_abierta = max(self.max_abierta, len(self.abierta))
        return fv, hv

    def es_obsoleta(self, ent):
        n, g = ent[3], ent[4]
        return n in self.cerrados or g > self.g.get(n, math.inf) + 1e-12

    def reconstruir(self, n):
        ruta = []
        while n is not None:
            ruta.append(n)
            n = self.padre[n]
        return ruta[::-1]

    def km_ruta(self):
        return sum(self.G.longitud(a, b) for a, b in zip(self.ruta, self.ruta[1:]))

    def eventos(self):
        G = self.G
        s0 = G.origen
        self.g[s0] = 0.0
        self.padre[s0] = None
        self._push(s0, 0.0)

        while True:
            vacia = not self.abierta
            yield ("vacia", vacia)
            if vacia:
                yield ("fallo",)
                return

            fv, _, _, n, gn = heapq.heappop(self.abierta)
            yield ("extraer", n, gn, fv)

            if self.es_obsoleta((fv, 0, 0, n, gn)):
                self.obsoletos += 1
                yield ("obsoleto", n, gn)
                continue
            yield ("vigente", n)

            es_meta = n == G.destino
            yield ("meta", n, es_meta)
            if es_meta:
                self.ruta = self.reconstruir(n)
                yield ("exito", n)
                return

            self.cerrados.add(n)
            self.expandidos += 1
            yield ("expandir", n)

            relajados = []
            for m in G.ady[n]:
                tipo = G.tipo(n, m)
                motivo = None
                if tipo == "cerrada":
                    motivo = "calle cerrada por obras"
                elif m in self.cerrados:
                    motivo = "cruce ya cerrado"
                yield ("probar", n, m, tipo, motivo)
                if motivo:
                    continue

                ng = self.g[n] + G.costo(n, m)
                if m in self.g and ng >= self.g[m] - 1e-12:
                    yield ("no_mejora", m, ng, self.g[m])
                    continue

                mejora = m in self.g
                self.g[m] = ng
                self.padre[m] = n
                fv2, hv2 = self._push(m, ng)
                relajados.append(m)
                yield ("encolar", m, ng, hv2, fv2, mejora, n)

            yield ("cerrar", n, relajados)

    def correr(self):
        ultimo = None
        for ev in self.eventos():
            ultimo = ev
        return ultimo[0] == "exito"


# ---------------------------------------------------------------------------
# Widgets planos (Label en vez de Button para respetar colores en macOS)
# ---------------------------------------------------------------------------

class Boton(tk.Label):
    ESTILOS = {
        "primary": (C["btn_primary"], "#0ea5e9"),
        "step": (C["btn_step"], "#818cf8"),
        "normal": (C["panel2"], "#162033"),
    }

    def __init__(self, parent, texto, cmd, tipo="normal"):
        self.bg, self.hover = self.ESTILOS[tipo]
        super().__init__(parent, text=texto, bg=self.bg, fg=C["text"], padx=10, pady=6,
                         font=(FONT, 9, "bold" if tipo != "normal" else "normal"), cursor="hand2",
                         highlightthickness=1, highlightbackground=C["border"])
        self.cmd = cmd
        self.activo = True
        self.bind("<Button-1>", lambda _e: self.activo and self.cmd())
        self.bind("<Enter>", lambda _e: self.activo and self.configure(bg=self.hover))
        self.bind("<Leave>", lambda _e: self.configure(bg=self.bg))

    def habilitar(self, si):
        self.activo = si
        self.configure(fg=C["text"] if si else C["dim"], cursor="hand2" if si else "arrow")


class Pildoras(tk.Frame):
    def __init__(self, parent, opciones, variable, comando=None):
        super().__init__(parent, bg=C["panel"])
        self.var, self.comando, self.items = variable, comando, {}
        for valor, texto in opciones:
            lb = tk.Label(self, text=texto, bg=C["panel2"], fg=C["muted"], padx=6, pady=4,
                          font=(FONT, 9), cursor="hand2", highlightthickness=1,
                          highlightbackground=C["border"])
            lb.pack(side="left", padx=(0, 4), fill="x", expand=True)
            lb.bind("<Button-1>", lambda _e, v=valor: self.elegir(v))
            self.items[valor] = lb
        self.refrescar()

    def elegir(self, v):
        if self.var.get() != v:
            self.var.set(v)
            self.refrescar()
            if self.comando:
                self.comando()

    def refrescar(self):
        for v, lb in self.items.items():
            on = v == self.var.get()
            lb.configure(bg=C["pill_on"] if on else C["panel2"], fg=C["accent"] if on else C["muted"],
                         highlightbackground=C["accent"] if on else C["border"],
                         font=(FONT, 9, "bold" if on else "normal"))


class Casilla(tk.Label):
    def __init__(self, parent, texto, variable, comando=None):
        super().__init__(parent, bg=C["panel"], fg=C["muted"], font=(FONT, 9), cursor="hand2",
                         anchor="w", justify="left", wraplength=260)
        self.texto, self.var, self.comando = texto, variable, comando
        self.bind("<Button-1>", self._clic)
        self.refrescar()

    def _clic(self, _e):
        self.var.set(not self.var.get())
        self.refrescar()
        if self.comando:
            self.comando()

    def refrescar(self):
        self.configure(text=("☑ " if self.var.get() else "☐ ") + self.texto,
                       fg=C["text"] if self.var.get() else C["muted"])


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------

ANCHO, ALTO = 680, 480
MARGEN = 30

FLUJO = [
    ("inicio", "1. INICIO", "ABIERTA ← {origen},  g = 0", "pill"),
    ("vacia", "2. ¿ABIERTA VACÍA?", "Sí → no hay ruta", "dec"),
    ("extraer", "3. EXTRAER MÍNIMO f", "f = g(n) + h(n)", "proc"),
    ("vigente", "4. ¿YA CERRADO?", "Sí → entrada obsoleta, se descarta", "dec"),
    ("meta", "5. ¿n ES EL DESTINO?", "Sí → paso 9", "dec"),
    ("expandir", "6. CERRAR n", "recorrer sus calles (aristas)", "proc"),
    ("probar", "7. ¿ABIERTA Y MEJORA g?", "g' = g(n) + t(n, m)  <  g(m)", "dec"),
    ("encolar", "8. PADRE Y PUSH", "ABIERTA ← (f(m), m)", "proc"),
    ("ruta", "9. RECONSTRUIR RUTA", "seguir punteros padre", "pill"),
]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Ruta más rápida · Dijkstra, A*, A* ponderada")
        root.configure(bg=C["bg"])
        root.geometry("1500x940")
        root.minsize(1380, 880)

        self.alg = tk.StringVar(value="DIJ")
        self.heur = tk.StringVar(value="L2")
        self.tam = tk.StringVar(value="8x6")
        self.modo = tk.StringVar(value="trafico")
        self.peso = tk.DoubleVar(value=2.0)
        self.velocidad = tk.IntVar(value=120)
        self.rotar = tk.BooleanVar(value=True)
        self.ver_costos = tk.BooleanVar(value=True)
        self.ver_g = tk.BooleanVar(value=True)

        self.G = mapa_ciudad(8, 6)
        self.resultados = {}
        self.corriendo = False
        self.terminado = False
        self.timer = None
        self.timer_ronda = None

        self._estilos()
        self._construir()
        self.reiniciar()

    # ------------------------------------------------------------ estilos

    def _estilos(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Treeview", font=(MONO, 9), rowheight=21, background=C["panel2"],
                     fieldbackground=C["panel2"], foreground=C["text"], borderwidth=0)
        st.configure("Treeview.Heading", font=(FONT, 8, "bold"), background=C["panel"],
                     foreground=C["accent"], relief="flat", borderwidth=0)
        st.map("Treeview.Heading", background=[("active", C["panel"])])
        st.map("Treeview", background=[("selected", C["pill_on"])], foreground=[("selected", C["text"])])
        st.configure("TNotebook", background=C["panel"], borderwidth=0)
        st.configure("TNotebook.Tab", background=C["panel2"], foreground=C["muted"],
                     padding=(12, 5), font=(FONT, 9), borderwidth=0)
        st.map("TNotebook.Tab", background=[("selected", C["pill_on"])],
               foreground=[("selected", C["accent"])])
        st.configure("Vertical.TScrollbar", background=C["panel"], troughcolor=C["panel2"],
                     bordercolor=C["panel2"], arrowcolor=C["muted"])

    def _panel(self, parent, titulo=None):
        f = tk.Frame(parent, bg=C["panel"], highlightbackground=C["border"], highlightthickness=1)
        if titulo:
            tk.Label(f, text=titulo.upper(), bg=C["panel"], fg=C["accent"],
                     font=(FONT, 9, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
            tk.Frame(f, height=1, bg=C["border"]).pack(fill="x", padx=10, pady=(0, 6))
        return f

    def _rotulo(self, parent, texto):
        tk.Label(parent, text=texto, bg=C["panel"], fg=C["muted"], font=(FONT, 9)).pack(
            anchor="w", padx=10, pady=(6, 2))

    # ------------------------------------------------------------ construcción

    def _construir(self):
        r = self.root

        head = tk.Frame(r, bg=C["bg"])
        head.pack(fill="x", pady=(10, 4))
        tk.Label(head, text="Ruta más rápida en un mapa de calles: Dijkstra, A* y A* ponderada",
                 bg=C["bg"], fg=C["accent"], font=(FONT, 15, "bold")).pack()
        tk.Label(head, text="Vértices = cruces · Aristas = calles · Costo = tiempo de viaje (min) = longitud / velocidad",
                 bg=C["bg"], fg=C["muted"], font=(FONT, 9)).pack()

        ban = tk.Frame(r, bg="#0d1829", highlightbackground=C["accent"], highlightthickness=1)
        ban.pack(fill="x", padx=16, pady=(4, 10))
        self.msg = tk.Label(ban, text="", bg="#0d1829", fg="#e2e8f0", font=(FONT, 10), anchor="w",
                            justify="left", wraplength=1300)
        self.msg.pack(side="left", padx=12, pady=7, fill="x", expand=True)
        self.tag = tk.Label(ban, text="LISTO", bg=C["accent"], fg="#04101e", font=(FONT, 8, "bold"),
                            padx=8, pady=3, width=12)
        self.tag.pack(side="right", padx=10)

        main = tk.Frame(r, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        # Izquierda: flujo + telemetría
        izq = tk.Frame(main, bg=C["bg"])
        izq.pack(side="left", fill="y")
        fl = self._panel(izq, "Diagrama de flujo")
        fl.pack(fill="x")
        self.fc = tk.Canvas(fl, width=290, height=470, bg=C["panel"], highlightthickness=0)
        self.fc.pack(padx=6, pady=(0, 8))
        self._dibujar_flujo()
        te = self._panel(izq, "Telemetría")
        te.pack(fill="both", expand=True, pady=(10, 0))
        self.tele = tk.Label(te, text="", bg=C["panel"], fg="#cbd5e1", font=(MONO, 9),
                             justify="left", anchor="nw")
        self.tele.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # Centro: mapa + pestañas
        cen = tk.Frame(main, bg=C["bg"])
        cen.pack(side="left", fill="both", expand=True, padx=12)
        mp = tk.Frame(cen, bg=C["panel"], highlightbackground=C["border"], highlightthickness=2)
        mp.pack()
        self.cv = tk.Canvas(mp, width=ANCHO, height=ALTO, bg=C["ground"], highlightthickness=0)
        self.cv.pack(padx=6, pady=6)
        self.cv.bind("<Button-1>", self.clic_mapa)
        self.hint = tk.Label(cen, text="", bg=C["bg"], fg=C["muted"], font=(FONT, 8))
        self.hint.pack(pady=(3, 6))

        self.nb = ttk.Notebook(cen)
        self.nb.pack(fill="both", expand=True)
        self.tree_ab = self._tabla(self.nb, "Lista ABIERTA (cola de prioridad)",
                                   [("orden", "#", 36), ("nodo", "Cruce", 60), ("g", "g (min)", 70),
                                    ("h", "h (min)", 70), ("f", "f", 70), ("padre", "Viene de", 70),
                                    ("nota", "Nota", 240)])
        self.tree_tr = self._tabla(self.nb, "Traza",
                                   [("paso", "Paso", 45), ("nodo", "Cruce", 60), ("g", "g", 60),
                                    ("f", "f", 60), ("suc", "Vecinos encolados", 230),
                                    ("com", "Comentario", 260)])
        self.tree_cmp = self._tabla(self.nb, "Comparación",
                                    [("alg", "Algoritmo", 110), ("cfg", "h · w", 120),
                                     ("t", "Tiempo (min)", 85), ("km", "Km", 55), ("cuadras", "Calles", 55),
                                     ("exp", "Expandidos", 75), ("gen", "Generados", 75),
                                     ("max", "Máx. ABIERTA", 90), ("opt", "¿Óptimo?", 110)])
        for t in (self.tree_ab, self.tree_tr, self.tree_cmp):
            t.tag_configure("top", background=C["frontier"], foreground="#1a1200")
            t.tag_configure("obs", foreground=C["dim"])
            t.tag_configure("live", foreground=C["current"])
            t.tag_configure("win", background=C["flow_ok"], foreground="#d1fae5")
            t.tag_configure("bad", foreground=C["goal"])

        # Derecha: controles + leyenda
        der = tk.Frame(main, bg=C["bg"])
        der.pack(side="left", fill="y")
        ct = self._panel(der, "Controles")
        ct.pack(fill="x")

        self._rotulo(ct, "Algoritmo")
        self.p_alg = Pildoras(ct, ALGORITMOS, self.alg, self.reiniciar)
        self.p_alg.pack(fill="x", padx=10)

        self.lbl_peso = tk.Label(ct, text="", bg=C["panel"], fg=C["muted"], font=(FONT, 9))
        self.lbl_peso.pack(anchor="w", padx=10, pady=(8, 0))
        self.sc_peso = tk.Scale(ct, from_=1.0, to=5.0, resolution=0.5, orient="horizontal",
                                variable=self.peso, showvalue=False, command=lambda _v: self.cambio_peso(),
                                bg=C["panel"], troughcolor=C["panel2"], highlightthickness=0, bd=0,
                                sliderrelief="flat", activebackground=C["accent"])
        self.sc_peso.pack(fill="x", padx=10)

        self._rotulo(ct, "Heurística h(n) = distancia / 60 km/h")
        self.p_heur = Pildoras(ct, HEURISTICAS, self.heur, self.reiniciar)
        self.p_heur.pack(fill="x", padx=10)

        self._rotulo(ct, "Tamaño del mapa")
        self.p_tam = Pildoras(ct, TAMANOS, self.tam, self.nuevo_mapa)
        self.p_tam.pack(fill="x", padx=10)

        self._rotulo(ct, "Clic en el mapa")
        self.p_modo = Pildoras(ct, [("origen", "Origen"), ("destino", "Destino"), ("trafico", "Tipo de calle")],
                               self.modo, self._etq_hint)
        self.p_modo.pack(fill="x", padx=10)

        self.lbl_vel = tk.Label(ct, text="", bg=C["panel"], fg=C["muted"], font=(FONT, 9))
        self.lbl_vel.pack(anchor="w", padx=10, pady=(8, 0))
        tk.Scale(ct, from_=10, to=800, resolution=10, orient="horizontal", variable=self.velocidad,
                 showvalue=False, command=lambda _v: self._etq_vel(), bg=C["panel"],
                 troughcolor=C["panel2"], highlightthickness=0, bd=0, sliderrelief="flat",
                 activebackground=C["accent"]).pack(fill="x", padx=10)
        self._etq_vel()

        Casilla(ct, "Mostrar minutos en las calles", self.ver_costos, self.dibujar).pack(fill="x", padx=10, pady=(4, 0))
        Casilla(ct, "Mostrar g en los cruces", self.ver_g, self.dibujar).pack(fill="x", padx=10)
        Casilla(ct, "Rotación automática: Dijkstra → A* → A* ponderada → mapa nuevo",
                self.rotar).pack(fill="x", padx=10, pady=(0, 4))

        fila = tk.Frame(ct, bg=C["panel"])
        fila.pack(fill="x", padx=10, pady=(6, 3))
        self.b_auto = Boton(fila, "▶ Auto", self.alternar_auto, "primary")
        self.b_auto.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.b_paso = Boton(fila, "Decisión ⏭", self.micro_paso, "step")
        self.b_paso.pack(side="left", fill="x", expand=True)
        fila = tk.Frame(ct, bg=C["panel"])
        fila.pack(fill="x", padx=10, pady=3)
        Boton(fila, "↻ Mapa nuevo", self.nuevo_mapa).pack(side="left", fill="x", expand=True, padx=(0, 3))
        Boton(fila, "Limpiar", self.reiniciar).pack(side="left", fill="x", expand=True)
        fila = tk.Frame(ct, bg=C["panel"])
        fila.pack(fill="x", padx=10, pady=(3, 10))
        Boton(fila, "⚖ Comparar los 3", self.comparar).pack(side="left", fill="x", expand=True)

        lg = self._panel(der, "Significado de los colores")
        lg.pack(fill="both", expand=True, pady=(10, 0))
        rej = tk.Frame(lg, bg=C["panel"])
        rej.pack(fill="x", padx=10, pady=(0, 8))
        nodos = [(C["start"], "Origen"), (C["goal"], "Destino"), (C["current"], "Cruce activo n"),
                 (C["testing"], "Vecino en prueba"), (C["frontier"], "ABIERTA"), (C["visited"], "CERRADA")]
        calles = [("avenida", "Avenida 60 km/h"), ("calle", "Calle 30 km/h"), ("trafico", "Tráfico 12 km/h"),
                  ("cerrada", "Cerrada (obras)"), ("_tree", "Árbol de búsqueda"), ("_path", "Ruta final")]
        for i, (color, texto) in enumerate(nodos):
            f = tk.Frame(rej, bg=C["panel2"])
            f.grid(row=i, column=0, sticky="ew", pady=2, padx=(0, 3))
            ic = tk.Canvas(f, width=16, height=16, bg=C["panel2"], highlightthickness=0)
            ic.pack(side="left", padx=4, pady=3)
            ic.create_oval(2, 2, 14, 14, fill=color, outline="")
            tk.Label(f, text=texto, bg=C["panel2"], fg=C["text"], font=(FONT, 8)).pack(side="left")
        for i, (tipo, texto) in enumerate(calles):
            f = tk.Frame(rej, bg=C["panel2"])
            f.grid(row=i, column=1, sticky="ew", pady=2)
            ic = tk.Canvas(f, width=22, height=16, bg=C["panel2"], highlightthickness=0)
            ic.pack(side="left", padx=4, pady=3)
            if tipo == "_tree":
                ic.create_line(2, 8, 20, 8, fill=C["tree"], width=2)
            elif tipo == "_path":
                ic.create_line(2, 8, 20, 8, fill=C["path"], width=5)
            else:
                _, _, gro, col = CALLES[tipo]
                ic.create_line(2, 8, 20, 8, fill=col, width=min(gro, 7),
                               dash=(3, 2) if tipo == "cerrada" else None)
            tk.Label(f, text=texto, bg=C["panel2"], fg=C["text"], font=(FONT, 8)).pack(side="left")
        rej.columnconfigure(0, weight=1)
        rej.columnconfigure(1, weight=1)
        tk.Label(lg, text="Río: solo se cruza por los puentes.", bg=C["panel"], fg=C["muted"],
                 font=(FONT, 8)).pack(anchor="w", padx=10, pady=(0, 8))

    def _tabla(self, nb, titulo, columnas):
        marco = tk.Frame(nb, bg=C["panel"])
        nb.add(marco, text=titulo)
        tree = ttk.Treeview(marco, columns=[c[0] for c in columnas], show="headings", height=7)
        for cid, texto, ancho in columnas:
            tree.heading(cid, text=texto, anchor="w")
            tree.column(cid, width=ancho, anchor="w", stretch=cid in ("nota", "com", "opt"))
        sb = ttk.Scrollbar(marco, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb.pack(side="right", fill="y", pady=6)
        return tree

    def _etq_vel(self):
        self.lbl_vel.configure(text=f"Velocidad: {self.velocidad.get()} ms por decisión")

    def _etq_hint(self):
        txt = {
            "origen": "Clic en un cruce para hacerlo ORIGEN.",
            "destino": "Clic en un cruce para hacerlo DESTINO.",
            "trafico": "Clic en una calle: avenida → calle → tráfico → cerrada → avenida.",
        }[self.modo.get()]
        self.hint.configure(text=txt)

    # ------------------------------------------------------------ diagrama de flujo

    def _dibujar_flujo(self):
        fc = self.fc
        self.flujo_ids, self.flujo_sub = {}, {}
        x0, x1, h, gap = 22, 252, 38, 13
        cx = (x0 + x1) / 2
        ycen = lambda i: 6 + i * (h + gap) + h / 2

        for i, (fid, titulo, sub, forma) in enumerate(FLUJO):
            y0 = 6 + i * (h + gap)
            y1, ym = y0 + h, y0 + h / 2
            if forma == "dec":
                pts = (x0 + 12, y0, x1 - 12, y0, x1, ym, x1 - 12, y1, x0 + 12, y1, x0, ym)
            elif forma == "pill":
                rr, pts = h / 2, []
                for k in range(0, 181, 15):
                    a = math.radians(-90 + k)
                    pts += [x1 - rr + rr * math.cos(a), ym + rr * math.sin(a)]
                for k in range(0, 181, 15):
                    a = math.radians(90 + k)
                    pts += [x0 + rr + rr * math.cos(a), ym + rr * math.sin(a)]
            else:
                pts = (x0, y0, x1, y0, x1, y1, x0, y1)
            self.flujo_ids[fid] = fc.create_polygon(*pts, fill=C["flow_box"], outline=C["flow_line"],
                                                    width=2, dash=(4, 2) if forma == "dec" else None)
            fc.create_text(cx, ym - 7, text=titulo, font=(FONT, 8, "bold"), fill=C["text"])
            self.flujo_sub[fid] = fc.create_text(cx, ym + 8, text=sub, font=(FONT, 8), fill=C["muted"])
            if i < len(FLUJO) - 1 and fid != "encolar":
                fc.create_line(cx, y1 + 1, cx, y1 + gap - 1, arrow="last", fill=C["dim"])
                etq = {"vacia": "No", "vigente": "No", "meta": "No", "probar": "Sí"}.get(fid)
                if etq:
                    fc.create_text(cx + 14, y1 + gap / 2, text=etq, font=(FONT, 7), fill=C["dim"])

        xa = x1 + 14
        fc.create_line(x1, ycen(7), xa, ycen(7), xa, ycen(1), x1 + 1, ycen(1), arrow="last", fill=C["dim"])
        fc.create_text(xa + 9, (ycen(7) + ycen(1)) / 2, text="ciclo", angle=90, font=(FONT, 7), fill=C["dim"])
        xb = x0 - 12
        fc.create_line(x0, ycen(3), xb, ycen(3), xb, ycen(1), x0 - 1, ycen(1), arrow="last",
                       fill=C["dim"], dash=(2, 2))
        xc = x0 - 6
        fc.create_line(x0, ycen(6), xc, ycen(6), xc, ycen(5), x0 - 1, ycen(5), arrow="last",
                       fill=C["dim"], dash=(2, 2))

    def resaltar(self, fid, tipo="activo"):
        for item in self.flujo_ids.values():
            self.fc.itemconfigure(item, fill=C["flow_box"], outline=C["flow_line"])
        if fid:
            fill, out = {"activo": (C["flow_active"], C["accent"]),
                         "exito": (C["flow_ok"], C["path"]),
                         "fallo": (C["flow_bad"], C["goal"])}[tipo]
            self.fc.itemconfigure(self.flujo_ids[fid], fill=fill, outline=out)

    # ------------------------------------------------------------ estado

    def cambio_peso(self):
        if self.alg.get() == "WAS":
            self.reiniciar()
        else:
            self._etq_peso()

    def _etq_peso(self):
        activo = self.alg.get() == "WAS"
        self.lbl_peso.configure(
            text=f"Peso w = {self.peso.get():.1f}" + ("" if activo else "   (solo A* ponderada)"),
            fg=C["text"] if activo else C["dim"])
        self.sc_peso.configure(state="normal" if activo else "disabled",
                               troughcolor=C["panel2"] if activo else C["panel"])

    def reiniciar(self):
        if self.timer_ronda:
            self.root.after_cancel(self.timer_ronda)
            self.timer_ronda = None
        self.detener()
        self.terminado = False
        self.b = Buscador(self.G, self.alg.get(), self.peso.get(), self.heur.get())
        self.gen = self.b.eventos()
        self.actual = None
        self.probando = None
        self.fila_viva = None
        self.sucesores = []
        self.paso_traza = 0

        for p in (self.p_alg, self.p_heur, self.p_tam, self.p_modo):
            p.refrescar()
        self._etq_peso()
        self._etq_hint()
        self.fc.itemconfigure(self.flujo_sub["extraer"], text=FORMULA[self.alg.get()])
        self._transformar()

        self.tree_tr.delete(*self.tree_tr.get_children())
        self.tree_tr.insert("", "end", values=(0, "—", "—", "—", self.G.origen, "Origen en ABIERTA con g = 0."))

        self.resaltar("inicio")
        txt = NOMBRE_ALG[self.alg.get()]
        if self.alg.get() != "DIJ":
            txt += f" con h = {NOMBRE_HEUR[self.heur.get()]}"
        if self.alg.get() == "WAS":
            txt += f" y w = {self.peso.get():.1f}"
        self.banner(f"Listo: {txt}. Buscar la ruta más rápida de {self.G.origen} a {self.G.destino}. "
                    "Presiona ▶ Auto o avanza con Decisión ⏭.", "INICIO")
        self.b_auto.habilitar(True)
        self.b_paso.habilitar(True)
        self._mostrar_abierta(inicial=True)
        self._telemetria()
        self._comparacion()
        self.dibujar()

    def cambiar_mapa(self, grafo):
        self.G = grafo
        self.resultados = {}
        self.reiniciar()

    def nuevo_mapa(self):
        c, f = map(int, self.tam.get().split("x"))
        self.cambiar_mapa(mapa_ciudad(c, f))
        self.banner(f"Mapa nuevo: {len(self.G.nodos)} cruces y {len(self.G.aristas)} calles. "
                    f"Ruta de {self.G.origen} a {self.G.destino}. Presiona ▶ Auto o Decisión ⏭.", "NUEVO")

    def banner(self, texto, tag, color=None):
        self.msg.configure(text=texto)
        self.tag.configure(text=tag, bg=color or C["accent"])

    # ------------------------------------------------------------ dibujo del mapa

    def _transformar(self):
        xs = [p[0] for p in self.G.nodos.values()]
        ys = [p[1] for p in self.G.nodos.values()]
        self.minx, self.miny = min(xs), min(ys)
        w, h = max(xs) - self.minx, max(ys) - self.miny
        self.esc = min((ANCHO - 2 * MARGEN) / w, (ALTO - 2 * MARGEN) / h)
        self.ox = (ANCHO - w * self.esc) / 2
        self.oy = (ALTO - h * self.esc) / 2
        self.radio = max(9, min(14, self.esc * 0.075))

    def px(self, n_o_punto):
        x, y = self.G.nodos[n_o_punto] if isinstance(n_o_punto, str) else n_o_punto
        return self.ox + (x - self.minx) * self.esc, self.oy + (y - self.miny) * self.esc

    def _etq(self, x, y, texto, fg, anchor="center", fuente=None, bg=None):
        t = self.cv.create_text(x, y, text=texto, fill=fg, anchor=anchor, font=fuente or (MONO, 7))
        x0, y0, x1, y1 = self.cv.bbox(t)
        rect = self.cv.create_rectangle(x0 - 2, y0, x1 + 2, y1, fill=bg or C["ground"], outline="")
        self.cv.tag_lower(rect, t)

    def dibujar(self):
        cv, G, b = self.cv, self.G, self.b
        cv.delete("all")

        # Río
        if G.rio:
            yc, semi, xa, xb, fase = G.rio
            arriba, abajo = [], []
            pasos = 40
            for k in range(pasos + 1):
                x = xa + (xb - xa) * k / pasos
                y = yc + math.sin(fase + k / pasos * 9) * semi * 0.6
                arriba.append(self.px((x, y - semi)))
                abajo.append(self.px((x, y + semi)))
            puntos = [c for p in arriba + abajo[::-1] for c in p]
            cv.create_polygon(*puntos, fill=C["river"], outline=C["river_edge"], smooth=True)
            px0, py0 = self.px((xa + 0.15, yc))
            cv.create_text(px0, py0, text="Río", fill="#5b8fc7", font=(FONT, 8, "italic"), anchor="w")

        # Calles base
        for par, tipo in G.aristas.items():
            a, b_ = tuple(par)
            (x1, y1), (x2, y2) = self.px(a), self.px(b_)
            _, _, gro, col = CALLES[tipo]
            cv.create_line(x1, y1, x2, y2, fill=col, width=gro, capstyle="round",
                           dash=(5, 4) if tipo == "cerrada" else None)
            if tipo == "avenida":
                cv.create_line(x1, y1, x2, y2, fill="#64748b", width=1, dash=(6, 6))
            if tipo == "cerrada":
                cv.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="✕", fill=C["goal"], font=(FONT, 9, "bold"))

        # Nombres de avenidas
        for texto, a, b_ in G.etiquetas:
            (x1, y1), (x2, y2) = self.px(a), self.px(b_)
            ang = -math.degrees(math.atan2(y2 - y1, x2 - x1))
            if ang < -90 or ang > 90:
                ang += 180
            cv.create_text((x1 + x2) / 2 + 0, (y1 + y2) / 2 - 10, text=texto, angle=ang,
                           fill="#7c8aa3", font=(FONT, 7, "italic"))

        # Árbol de búsqueda (punteros padre)
        for n, p in b.padre.items():
            if p is not None:
                (x1, y1), (x2, y2) = self.px(p), self.px(n)
                cv.create_line(x1, y1, x2, y2, fill=C["tree"], width=2)

        # Calle en prueba
        if self.actual and self.probando:
            (x1, y1), (x2, y2) = self.px(self.actual), self.px(self.probando)
            cv.create_line(x1, y1, x2, y2, fill=C["testing"], width=5, capstyle="round")

        # Ruta final
        if len(b.ruta) > 1:
            pts = [c for n in b.ruta for c in self.px(n)]
            cv.create_line(*pts, fill=C["path"], width=6, capstyle="round", joinstyle="round")

        # Minutos por calle
        if self.ver_costos.get():
            for par, tipo in G.aristas.items():
                if tipo == "cerrada":
                    continue
                a, b_ = tuple(par)
                (x1, y1), (x2, y2) = self.px(a), self.px(b_)
                self._etq((x1 + x2) / 2, (y1 + y2) / 2, num(G.costo(a, b_), 1),
                          "#fda4af" if tipo == "trafico" else "#cbd5e1")

        # Cruces
        en_abierta = {e[3] for e in b.abierta if not b.es_obsoleta(e)}
        r = self.radio
        for n in G.nodos:
            x, y = self.px(n)
            oscuro = False
            if n == G.origen:
                fill, oscuro = C["start"], True
            elif n == G.destino:
                fill = C["goal"]
            elif n == self.actual:
                fill = C["current"]
            elif n == self.probando:
                fill = C["testing"]
            elif n in b.ruta:
                fill, oscuro = C["path"], True
            elif n in b.cerrados:
                fill = C["visited"]
            elif n in en_abierta:
                fill, oscuro = C["frontier"], True
            else:
                fill = C["node"]
            borde = C["current"] if n == self.actual else C["ground"]
            cv.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=borde, width=3 if n == self.actual else 2)
            cv.create_text(x, y, text=n, fill="#0b1120" if oscuro else C["text"], font=(FONT, 7, "bold"))
            if self.ver_g.get() and n in b.g:
                self._etq(x + r - 2, y - r - 1, f"g={num(b.g[n], 1)}", C["accent"], anchor="sw",
                          fuente=(MONO, 7, "bold"), bg=C["panel2"])

    # ------------------------------------------------------------ paneles

    def _mostrar_abierta(self, inicial=False):
        t, b = self.tree_ab, self.b
        t.delete(*t.get_children())
        if inicial:
            s0 = self.G.origen
            t.insert("", "end", tags=("top",), values=(1, s0, 0, num(b.h(s0)), num(b.f(s0, 0)), "—", "sale primero"))
            return
        primero = True
        for i, ent in enumerate(sorted(b.abierta), 1):
            fv, _, _, n, g = ent
            obs = b.es_obsoleta(ent)
            if obs:
                tag, nota = "obs", "obsoleta (ya cerrado o hay mejor g)"
            elif primero:
                tag, nota, primero = "top", "sale primero", False
            else:
                tag, nota = "", ""
            t.insert("", "end", tags=(tag,), values=(i, n, num(g), num(b.h(n)), num(fv),
                                                     b.padre.get(n) or "—", nota))

    def _telemetria(self, n=None):
        b = self.b
        n = n or self.actual or self.G.origen
        g = b.g.get(n)
        lineas = [
            f"Algoritmo    {NOMBRE_ALG[b.algoritmo]}",
            f"Fórmula      {FORMULA[b.algoritmo]}",
            f"Heurística   {NOMBRE_HEUR[b.heur] if b.peso > 0 else '— (no usa)'}",
            f"Peso w       {num(b.peso)}",
            "─" * 27,
            f"Cruce n      {n}",
            f"g(n)         {num(g)} min",
            f"h(n)         {num(b.h(n))} min",
            f"f(n)         {num(b.f(n, g)) if g is not None else '—'}",
            f"Línea recta  {num(b.distancia(n, 'L2'))} km",
            "─" * 27,
            f"ABIERTA      {sum(1 for x in b.abierta if not b.es_obsoleta(x))} vigentes",
            f"CERRADA      {len(b.cerrados)} de {len(self.G.nodos)}",
            f"Expandidos   {b.expandidos}",
            f"Generados    {b.generados}",
            f"Obsoletos    {b.obsoletos}",
            f"Máx ABIERTA  {b.max_abierta}",
        ]
        self.tele.configure(text="\n".join(lineas))

    def _comparacion(self):
        t = self.tree_cmp
        t.delete(*t.get_children())
        ref = self.resultados.get("DIJ", {}).get("t")
        for clave, nombre in ALGORITMOS:
            r = self.resultados.get(clave)
            if not r:
                t.insert("", "end", tags=("obs",), values=(nombre, "", "", "", "", "", "", "", "sin correr"))
                continue
            if r["t"] is None:
                opt, tag = "sin ruta", "bad"
            elif ref is None:
                opt, tag = "corre Dijkstra", ""
            elif abs(r["t"] - ref) < 1e-9:
                opt, tag = "Sí", "win"
            else:
                opt, tag = f"No (+{num(r['t'] - ref)} min)", "bad"
            t.insert("", "end", tags=(tag,), values=(nombre, r["cfg"], num(r["t"]), r["km"], r["cuadras"],
                                                     r["exp"], r["gen"], r["max"], opt))

    def _guardar_resultado(self, b):
        ok = bool(b.ruta)
        self.resultados[b.algoritmo] = {
            "cfg": "—" if b.peso == 0 else NOMBRE_HEUR[b.heur] + (f" · w={num(b.peso)}" if b.algoritmo == "WAS" else ""),
            "t": b.g.get(self.G.destino) if ok else None,
            "km": num(b.km_ruta()) if ok else "—",
            "cuadras": len(b.ruta) - 1 if ok else "—",
            "exp": b.expandidos, "gen": b.generados, "max": b.max_abierta,
        }
        self._comparacion()

    def comparar(self):
        for clave, _ in ALGORITMOS:
            b = Buscador(self.G, clave, self.peso.get(), self.heur.get())
            b.correr()
            self._guardar_resultado(b)
        self.nb.select(2)
        r = self.resultados
        self.banner("Comparación en este mapa · " + "  ·  ".join(
            f"{NOMBRE_ALG[k]}: {num(r[k]['t'])} min, {r[k]['exp']} expandidos" for k, _ in ALGORITMOS),
            "COMPARACIÓN")

    # ------------------------------------------------------------ máquina de decisiones

    def micro_paso(self):
        if self.terminado:
            return
        try:
            ev = next(self.gen)
        except StopIteration:
            self.terminado = True
            return
        b, G, tipo = self.b, self.G, ev[0]

        if tipo == "vacia":
            self.resaltar("vacia")
            self.probando = None
            if not ev[1]:
                vig = sum(1 for x in b.abierta if not b.es_obsoleta(x))
                self.banner(f"Paso 2: ¿ABIERTA vacía? NO ({len(b.abierta)} entradas, {vig} vigentes).", "DECISIÓN")

        elif tipo == "fallo":
            self.resaltar("vacia", "fallo")
            self.banner(f"ABIERTA vacía: no hay forma de llegar de {G.origen} a {G.destino} "
                        "(¿calles cerradas?).", "SIN RUTA", C["goal"])
            self._guardar_resultado(b)
            self.finalizar()

        elif tipo == "extraer":
            _, n, g, fv = ev
            self.resaltar("extraer")
            self.actual = n
            self.banner(f"Paso 3: Se extrae el cruce {n} con el menor f = {num(fv)}  "
                        f"(g = {num(g)} min, h = {num(b.h(n))} min).", "ACTIVO")
            self._mostrar_abierta()
            self._telemetria(n)

        elif tipo == "obsoleto":
            _, n, g = ev
            self.resaltar("vigente")
            fila = self.tree_tr.insert("", "end", tags=("obs",), values=(
                "·", n, num(g), "", "—", "Entrada obsoleta: ya se cerró con mejor g. Se descarta."))
            self.tree_tr.see(fila)
            self.banner(f"Paso 4: {n} ya está CERRADO. Esta entrada vieja (g = {num(g)}) se descarta.", "OBSOLETO")
            self.actual = None

        elif tipo == "vigente":
            self.resaltar("vigente")
            self.banner(f"Paso 4: ¿{ev[1]} ya está cerrado? NO, la entrada es vigente.", "DECISIÓN")

        elif tipo == "meta":
            _, n, es = ev
            self.resaltar("meta")
            if not es:
                self.banner(f"Paso 5: ¿{n} es el destino {G.destino}? NO. Se cierra y se recorren sus calles.",
                            "DECISIÓN")

        elif tipo == "expandir":
            n = ev[1]
            self.resaltar("expandir")
            self.paso_traza += 1
            self.sucesores = []
            self.fila_viva = self.tree_tr.insert("", "end", tags=("live",), values=(
                self.paso_traza, n, num(b.g[n]), num(b.f(n, b.g[n])), "", "En expansión…"))
            self.tree_tr.see(self.fila_viva)
            self.banner(f"Paso 6: {n} pasa a CERRADA (morado). Tiene {len(G.ady[n])} calles: "
                        f"{', '.join(G.ady[n])}.", "EXPANSIÓN")
            self._telemetria(n)

        elif tipo == "probar":
            _, n, m, tc, motivo = ev
            self.resaltar("probar")
            self.probando = m
            nombre, vel, _, _ = CALLES[tc]
            if motivo:
                self.banner(f"Paso 7: {n} → {m} ({nombre.lower()}): ❌ DESCARTADA ({motivo}).", "FILTRADO")
            else:
                km, t = G.longitud(n, m), G.costo(n, m)
                ng = b.g[n] + t
                previo = b.g.get(m)
                extra = f"  ·  g({m}) actual = {num(previo)}" if previo is not None else "  ·  cruce nuevo"
                self.banner(f"Paso 7: {n} → {m}: {nombre.lower()} de {km:.2f} km a {vel} km/h = {num(t)} min.  "
                            f"g' = {num(b.g[n])} + {num(t)} = {num(ng)}{extra}", "RELAJACIÓN")

        elif tipo == "no_mejora":
            _, m, ng, gv = ev
            self.banner(f"Paso 7: {m} ya tiene g = {num(gv)} ≤ {num(ng)}. Este camino NO mejora, se ignora.",
                        "SIN MEJORA")

        elif tipo == "encolar":
            _, m, ng, hv, fv, mejora, padre = ev
            self.resaltar("encolar")
            self.probando = None
            self.sucesores.append(m + ("*" if mejora else ""))
            vals = list(self.tree_tr.item(self.fila_viva, "values"))
            vals[4] = " ".join(self.sucesores)
            self.tree_tr.item(self.fila_viva, values=vals)
            accion = "Camino MÁS RÁPIDO encontrado: se cambia su padre" if mejora else "Entra a ABIERTA"
            self.banner(f"Paso 8: {m}.padre = {padre},  g = {num(ng)},  h = {num(hv)},  f = {num(fv)}. {accion}.",
                        "ENCOLADO")
            self._mostrar_abierta()
            self._telemetria(padre)

        elif tipo == "cerrar":
            _, n, relajados = ev
            self.actual = None
            k = len(relajados)
            com = ("Sin vecinos nuevos o mejores." if k == 0 else
                   f"{k} vecino(s) encolado(s)." + ("  * = mejora de g" if any("*" in s for s in self.sucesores) else ""))
            vals = list(self.tree_tr.item(self.fila_viva, "values"))
            vals[4] = vals[4] or "—"
            vals[5] = com
            self.tree_tr.item(self.fila_viva, values=vals, tags=())
            self.banner(f"Calles de {n} revisadas. Se vuelve a la cola de prioridad.", "CERRADO")

        elif tipo == "exito":
            n = ev[1]
            self.resaltar("ruta", "exito")
            self.actual = None
            self.probando = None
            self.paso_traza += 1
            fila = self.tree_tr.insert("", "end", tags=("win",), values=(
                self.paso_traza, n, num(b.g[n]), num(b.f(n, b.g[n])), "—", "¡Destino extraído! Fin."))
            self.tree_tr.see(fila)
            self._guardar_resultado(b)
            ref = self.resultados.get("DIJ", {}).get("t")
            nota = ""
            if ref is not None and b.algoritmo != "DIJ":
                nota = "  ·  Óptima ✔" if abs(b.g[n] - ref) < 1e-9 else f"  ·  {num(b.g[n] - ref)} min más lenta que Dijkstra"
            self.banner(f"¡DESTINO ALCANZADO! {' → '.join(b.ruta)}  ·  {num(b.g[n])} min, {num(b.km_ruta())} km, "
                        f"{b.expandidos} cruces expandidos.{nota}", "ÉXITO", C["path"])
            self._mostrar_abierta()
            self.finalizar()

        self.dibujar()

    def finalizar(self):
        seguia = self.corriendo
        self.terminado = True
        self.detener()
        self.b_auto.habilitar(False)
        self.b_paso.habilitar(False)
        if self.rotar.get():
            orden = [k for k, _ in ALGORITMOS]
            i = orden.index(self.alg.get())
            if i < len(orden) - 1:
                sig = f"sigue {NOMBRE_ALG[orden[i + 1]]} en el mismo mapa"
            else:
                sig = "sigue un mapa nuevo"
                self.nb.select(2)
            self.msg.configure(text=self.msg.cget("text") + f"   →  En 3 s {sig}…")
            self.timer_ronda = self.root.after(3000, lambda: self.siguiente_ronda(seguia))

    def siguiente_ronda(self, seguir_auto):
        self.timer_ronda = None
        orden = [k for k, _ in ALGORITMOS]
        i = orden.index(self.alg.get())
        if i < len(orden) - 1:
            self.alg.set(orden[i + 1])
            self.reiniciar()
        else:
            self.alg.set(orden[0])
            self.nuevo_mapa()
        self.nb.select(0)
        if seguir_auto:
            self.alternar_auto()

    # ------------------------------------------------------------ automático

    def alternar_auto(self):
        if self.corriendo:
            self.detener()
        elif not self.terminado:
            self.corriendo = True
            self.b_auto.configure(text="⏸ Pausar")
            self.b_paso.habilitar(False)
            self.bucle()

    def detener(self):
        self.corriendo = False
        if self.timer:
            self.root.after_cancel(self.timer)
            self.timer = None
        if hasattr(self, "b_auto"):
            self.b_auto.configure(text="▶ Auto")
            self.b_paso.habilitar(not self.terminado)

    def bucle(self):
        if not self.corriendo or self.terminado:
            return
        self.micro_paso()
        self.timer = self.root.after(self.velocidad.get(), self.bucle)

    # ------------------------------------------------------------ edición con clic

    def clic_mapa(self, ev):
        G = self.G
        cerca, dmin = None, 1e9
        for n in G.nodos:
            x, y = self.px(n)
            d = math.hypot(ev.x - x, ev.y - y)
            if d < dmin:
                cerca, dmin = n, d
        modo = self.modo.get()

        if modo in ("origen", "destino"):
            if dmin > self.radio + 6:
                return
            otro = G.destino if modo == "origen" else G.origen
            if cerca == otro:
                return
            nuevo = G.copia()
            setattr(nuevo, modo, cerca)
            self.cambiar_mapa(nuevo)
            self.banner(f"{'Origen' if modo == 'origen' else 'Destino'} cambiado a {cerca}. Búsqueda reiniciada.",
                        "EDITADO")
            return

        # Modo tipo de calle: arista más cercana al clic
        mejor, dmin = None, 1e9
        for par in G.aristas:
            a, b_ = tuple(par)
            (x1, y1), (x2, y2) = self.px(a), self.px(b_)
            dx, dy = x2 - x1, y2 - y1
            t = max(0, min(1, ((ev.x - x1) * dx + (ev.y - y1) * dy) / (dx * dx + dy * dy or 1)))
            d = math.hypot(ev.x - (x1 + t * dx), ev.y - (y1 + t * dy))
            if d < dmin:
                mejor, dmin = par, d
        if mejor is None or dmin > 9:
            return
        nuevo = G.copia()
        nuevo.aristas[mejor] = CICLO_TIPO[G.aristas[mejor]]
        nuevo._ady()
        a, b_ = sorted(mejor)
        self.cambiar_mapa(nuevo)
        self.banner(f"Calle {a}–{b_} ahora es: {CALLES[nuevo.aristas[mejor]][0].lower()}. Búsqueda reiniciada.",
                    "EDITADO")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()