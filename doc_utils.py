def map_sections(parsed_html):
    mapped = {
            str(i):
            {
                "content":parsed_html[i]
            } for i in range(len(parsed_html))
        }
    return mapped