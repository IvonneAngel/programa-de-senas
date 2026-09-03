from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


SENTENCE_TARGET_COUNT: Final = 700
SENTENCE_MODEL_STATUS: Final = "objetivo_no_entrenado"
SENTENCE_VALIDATION_STATUS: Final = "necesita_validacion_lsm_nativa"


@dataclass(frozen=True, slots=True)
class SentenceTarget:
    target_id: str
    domain: str
    spanish: str
    rough_lsm_gloss: str
    model_status: str = SENTENCE_MODEL_STATUS
    validation_status: str = SENTENCE_VALIDATION_STATUS
    recognition_unit: str = "sentence_or_intent"
    minimum_sequences: int = 20
    strong_sequences: int = 50


DOMAIN_BLUEPRINTS: tuple[dict[str, object], ...] = (
    {
        "domain": "saludos",
        "topics": (
            "hola",
            "buenos dias",
            "buenas tardes",
            "buenas noches",
            "gracias",
            "por favor",
            "perdon",
            "nos vemos",
            "bienvenido",
            "hasta manana",
        ),
        "templates": (
            ("quiero decir {topic}", "yo decir {topic} querer"),
            ("necesito usar {topic}", "yo {topic} necesitar"),
            ("voy a senar {topic}", "yo {topic} senar ir"),
            ("aprendo la frase {topic}", "yo frase {topic} aprender"),
            ("practico la frase {topic}", "yo frase {topic} practicar"),
            ("entiendo la frase {topic}", "yo frase {topic} entender"),
            ("repito la frase {topic}", "yo frase {topic} repetir"),
        ),
    },
    {
        "domain": "familiares",
        "topics": (
            "mama",
            "papa",
            "hermana",
            "hermano",
            "abuela",
            "abuelo",
            "hija",
            "hijo",
            "familia",
            "amigo",
        ),
        "templates": (
            ("quiero ver a mi {topic}", "yo mi {topic} ver querer"),
            ("necesito hablar con mi {topic}", "yo mi {topic} hablar necesitar"),
            ("busco a mi {topic}", "yo mi {topic} buscar"),
            ("mi {topic} esta en casa", "mi {topic} casa estar"),
            ("mi {topic} va a llegar", "mi {topic} llegar futuro"),
            ("ayudo a mi {topic}", "yo mi {topic} ayudar"),
            ("mi {topic} me ayuda", "mi {topic} yo ayudar"),
        ),
    },
    {
        "domain": "escuela",
        "topics": (
            "escuela",
            "tarea",
            "clase",
            "maestro",
            "libro",
            "cuaderno",
            "examen",
            "pregunta",
            "respuesta",
            "recreo",
        ),
        "templates": (
            ("voy a revisar {topic}", "yo {topic} revisar ir"),
            ("necesito la {topic}", "yo {topic} necesitar"),
            ("busco la {topic}", "yo {topic} buscar"),
            ("aprendo sobre {topic}", "yo {topic} aprender"),
            ("practico {topic}", "yo {topic} practicar"),
            ("entiendo {topic}", "yo {topic} entender"),
            ("no entiendo {topic}", "yo {topic} entender no"),
        ),
    },
    {
        "domain": "alimentos",
        "topics": (
            "agua",
            "comida",
            "tortilla",
            "pan",
            "fruta",
            "verdura",
            "leche",
            "cafe",
            "arroz",
            "sopa",
        ),
        "templates": (
            ("quiero {topic}", "yo {topic} querer"),
            ("necesito {topic}", "yo {topic} necesitar"),
            ("busco {topic}", "yo {topic} buscar"),
            ("me gusta {topic}", "yo {topic} gustar"),
            ("no quiero {topic}", "yo {topic} querer no"),
            ("voy a comprar {topic}", "yo {topic} comprar ir"),
            ("voy a preparar {topic}", "yo {topic} preparar ir"),
        ),
    },
    {
        "domain": "casa",
        "topics": (
            "casa",
            "cuarto",
            "cocina",
            "bano",
            "puerta",
            "ventana",
            "mesa",
            "silla",
            "cama",
            "ropa",
        ),
        "templates": (
            ("voy a revisar {topic}", "yo {topic} revisar ir"),
            ("necesito limpiar {topic}", "yo {topic} limpiar necesitar"),
            ("busco {topic}", "yo {topic} buscar"),
            ("{topic} esta aqui", "{topic} aqui estar"),
            ("{topic} esta alla", "{topic} alla estar"),
            ("quiero arreglar {topic}", "yo {topic} arreglar querer"),
            ("voy a usar {topic}", "yo {topic} usar ir"),
        ),
    },
    {
        "domain": "salud",
        "topics": (
            "doctor",
            "medicina",
            "dolor",
            "cita",
            "hospital",
            "enfermera",
            "fiebre",
            "tos",
            "descanso",
            "ayuda medica",
        ),
        "templates": (
            ("necesito {topic}", "yo {topic} necesitar"),
            ("busco {topic}", "yo {topic} buscar"),
            ("voy a revisar {topic}", "yo {topic} revisar ir"),
            ("tengo {topic}", "yo {topic} tener"),
            ("quiero explicar {topic}", "yo {topic} explicar querer"),
            ("necesito preguntar sobre {topic}", "yo {topic} preguntar necesitar"),
            ("necesito informacion sobre {topic}", "yo {topic} informacion necesitar"),
        ),
    },
    {
        "domain": "transporte",
        "topics": (
            "camion",
            "metro",
            "taxi",
            "carro",
            "bicicleta",
            "calle",
            "parada",
            "boleto",
            "direccion",
            "mapa",
        ),
        "templates": (
            ("voy en {topic}", "yo {topic} ir"),
            ("necesito {topic}", "yo {topic} necesitar"),
            ("busco {topic}", "yo {topic} buscar"),
            ("donde esta {topic}", "{topic} donde"),
            ("quiero pagar {topic}", "yo {topic} pagar querer"),
            ("voy a esperar {topic}", "yo {topic} esperar ir"),
            ("no encuentro {topic}", "yo {topic} encontrar no"),
        ),
    },
    {
        "domain": "trabajo",
        "topics": (
            "trabajo",
            "jefe",
            "reunion",
            "proyecto",
            "horario",
            "dinero",
            "documento",
            "mensaje",
            "computadora",
            "telefono",
        ),
        "templates": (
            ("voy a revisar {topic}", "yo {topic} revisar ir"),
            ("necesito {topic}", "yo {topic} necesitar"),
            ("busco {topic}", "yo {topic} buscar"),
            ("termino {topic}", "yo {topic} terminar"),
            ("explico {topic}", "yo {topic} explicar"),
            ("pregunto sobre {topic}", "yo {topic} preguntar"),
            ("no entiendo {topic}", "yo {topic} entender no"),
        ),
    },
    {
        "domain": "tiempo",
        "topics": (
            "hoy",
            "manana",
            "ayer",
            "ahora",
            "despues",
            "temprano",
            "tarde",
            "semana",
            "mes",
            "ano",
        ),
        "templates": (
            ("quiero hacerlo {topic}", "yo hacer {topic} querer"),
            ("necesito ir {topic}", "yo ir {topic} necesitar"),
            ("nos vemos {topic}", "nosotros ver {topic}"),
            ("trabajo {topic}", "yo trabajar {topic}"),
            ("estudio {topic}", "yo estudiar {topic}"),
            ("pregunto por {topic}", "yo {topic} preguntar"),
            ("no puedo {topic}", "yo poder {topic} no"),
        ),
    },
    {
        "domain": "emergencia",
        "topics": (
            "ayuda",
            "peligro",
            "policia",
            "ambulancia",
            "fuego",
            "perdido",
            "robo",
            "accidente",
            "dolor fuerte",
            "salida",
        ),
        "templates": (
            ("necesito {topic}", "yo {topic} necesitar"),
            ("hay {topic}", "{topic} haber"),
            ("busco {topic}", "yo {topic} buscar"),
            ("llama a {topic}", "{topic} llamar"),
            ("quiero reportar {topic}", "yo {topic} reportar querer"),
            ("donde esta {topic}", "{topic} donde"),
            ("no encuentro {topic}", "yo {topic} encontrar no"),
        ),
    },
)


def build_sentence_curriculum(limit: int = SENTENCE_TARGET_COUNT) -> tuple[SentenceTarget, ...]:
    """build sentence curriculum."""
    if limit < 1 or limit > SENTENCE_TARGET_COUNT:
        raise ValueError(f"limit must be between 1 and {SENTENCE_TARGET_COUNT}")

    targets: list[SentenceTarget] = []
    for blueprint in DOMAIN_BLUEPRINTS:
        domain = str(blueprint["domain"])
        topics = tuple(str(topic) for topic in blueprint["topics"])
        templates = tuple(blueprint["templates"])
        for template_index, raw_template in enumerate(templates):
            spanish_template, gloss_template = raw_template
            for topic_index, topic in enumerate(topics):
                target_number = len(targets) + 1
                targets.append(
                    SentenceTarget(
                        target_id=f"sent_{target_number:04d}",
                        domain=domain,
                        spanish=str(spanish_template).format(topic=topic),
                        rough_lsm_gloss=str(gloss_template).format(topic=topic),
                    )
                )
                if len(targets) == limit:
                    return tuple(targets)

    return tuple(targets)


def sentence_targets_as_dicts(limit: int = SENTENCE_TARGET_COUNT) -> list[dict[str, object]]:
    return [asdict(target) for target in build_sentence_curriculum(limit)]