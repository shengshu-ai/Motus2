"""Explicit opt-in Motus-style trunk/interface learning-rate groups."""


def parameter_groups(model, training):
    base_lr = float(training.learning_rate)
    wan_lr = float(training.get("wan_learning_rate", base_lr))
    scheme = training.get("parameter_group_scheme", "wan_other")
    groups = {"interface": [], "trunk": []}
    if scheme == "motus_interface":
        prefixes = ("action_expert.input_encoder.", "action_expert.projection.", "action_expert.decoder.")
        for name, param in model.named_parameters():
            if param.requires_grad:
                groups["interface" if name.startswith(prefixes) else "trunk"].append(param)
        if not all(groups.values()):
            raise ValueError("motus_interface requires both interface and trunk parameters")
        return [dict(params=groups["trunk"], lr=wan_lr, name="trunk"),
                dict(params=groups["interface"], lr=base_lr, name="action_interface")]
    if scheme != "wan_other":
        raise ValueError(f"Unknown parameter_group_scheme: {scheme}")
    wan = [p for p in model.video_model.wan_model.parameters() if p.requires_grad]
    wan_ids = {id(p) for p in wan}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in wan_ids]
    return ([dict(params=other, lr=base_lr)] if other else []) + (
        [dict(params=wan, lr=wan_lr)] if wan else [])
