    tracks = sorted({
        t
        for _, data in G.nodes(data=True)
        if data.get("track") != "None"
        for t in (data["track"] if isinstance(data["track"], list) else [data["track"]])
    })