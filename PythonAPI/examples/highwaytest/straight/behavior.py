"""Level-consistency behavior classification and matrix aggregation."""
CLASSES=('invalid','collision','strong','mild','none')


def classify(episode):
    """Map one evaluated episode to its actual behavior class, independent of training intent.

    Priority: invalid event > target collision > strong non-collision
    (qualified cut-in reaction or qualified stop) > mild blocking > no effect.
    """
    if episode.get('invalid'):return 'invalid'
    if episode.get('collision'):return 'collision'
    if episode.get('cut_in') or episode.get('stop_qualified'):return 'strong'
    if float(episode.get('blocking_time',0.))>=1.5:return 'mild'
    return 'none'


def build_matrix(entries):
    """Aggregate (level, method, seed, episodes) into intent x method rows.

    Cells are row-normalized episode fractions over the group's episodes.
    """
    groups={}
    for level,method,seed,episodes in entries:
        group=groups.setdefault((level,method),{'episodes':[],'seeds':set()})
        group['episodes'].extend(episodes);group['seeds'].add(seed)
    rows=[]
    for (level,method),group in sorted(groups.items()):
        counts={name:0 for name in CLASSES}
        for episode in group['episodes']:counts[classify(episode)]+=1
        total=len(group['episodes'])
        rows.append(dict(level=level,method=method,seeds=sorted(group['seeds']),
                         cases=total,classes={name:(counts[name]/total if total else 0.) for name in CLASSES}))
    return {'rows':rows}
