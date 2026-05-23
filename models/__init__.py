from .model import DecAlign

MODEL_MAP = {
    'decalign': DecAlign,
}


def build_model(args):
    if args.model_name not in MODEL_MAP:
        raise ValueError("Unsupported model name. The only supported model is 'decalign'.")
    return MODEL_MAP[args.model_name](args)


__all__ = ['DecAlign', 'build_model']
