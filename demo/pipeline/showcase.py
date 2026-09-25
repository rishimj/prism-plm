"""Curated proteins featured in the interactive explorer."""

SHOWCASE = [
    {"acc": "P01116", "short": "KRAS", "title": "GTPase KRas",
     "blurb": "A molecular on/off switch mutated in roughly one in four human cancers. Its P-loop grips GTP.",
     "focus": "binding"},
    {"acc": "P68871", "short": "HBB", "title": "Hemoglobin subunit beta",
     "blurb": "Carries oxygen in red blood cells. A single Glu6Val substitution causes sickle-cell disease.",
     "focus": "helix"},
    {"acc": "P42212", "short": "GFP", "title": "Green fluorescent protein",
     "blurb": "The jellyfish protein that glows green. An 11-stranded beta-barrel that became biology's favourite reporter.",
     "focus": "strand"},
    {"acc": "P04637", "short": "p53", "title": "Cellular tumor antigen p53",
     "blurb": "The 'guardian of the genome' and the most frequently mutated gene in human cancer.",
     "focus": "dna_binding"},
    {"acc": "P07550", "short": "ADRB2", "title": "Beta-2 adrenergic receptor",
     "blurb": "A G-protein-coupled receptor threaded seven times through the cell membrane; the target of asthma drugs.",
     "focus": "transmembrane"},
    {"acc": "P0DP23", "short": "CALM1", "title": "Calmodulin-1",
     "blurb": "The cell's calcium sensor. Four EF-hand loops each cradle a calcium ion.",
     "focus": "binding"},
    {"acc": "P61626", "short": "LYZ", "title": "Lysozyme C",
     "blurb": "An antibacterial enzyme in tears and saliva, held together by four disulfide bonds.",
     "focus": "disulfide"},
    {"acc": "P01308", "short": "INS", "title": "Insulin",
     "blurb": "The hormone that controls blood sugar, and the first protein ever sequenced (Sanger, 1955).",
     "focus": "disulfide"},
    {"acc": "Q9UK33", "short": "ZNF580", "title": "Zinc finger protein 580",
     "blurb": "A transcription factor whose C2H2 zinc fingers clamp a zinc ion to read DNA.",
     "focus": "zinc_finger"},
]

# Proteins used as steering targets (short enough for fast CPU inference)
STEERING_TARGETS = ["P42212", "P68871", "P61626", "P01116"]
