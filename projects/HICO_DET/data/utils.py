def get_hoi_box(human_bbox, object_bbox):
    h_xmin, h_ymin, h_xmax, h_ymax = human_bbox
    o_xmin, o_ymin, o_xmax, o_ymax = object_bbox
    xmin, ymin = min(h_xmin, o_xmin), min(h_ymin, o_ymin)
    xmax, ymax = max(h_xmax, o_xmax), max(h_ymax, o_ymax)
    hoi_bbox = [xmin, ymin, xmax, ymax]
    return hoi_bbox