from transformers import Trainer

class RegularizedTrainer(Trainer):
    def __init__(
        self,
        *args,
        regularized: bool = False,
        reg_fn = None,
        reg_kwargs: dict = None,
        reg_coeff: float = 1e-3,
        **kwargs
    ):
        super().__init__(*args,**kwargs)
        self.regularized = regularized
        if self.regularized:
            if reg_fn is None:
                raise ValueError("regularization must be non-empty!")
            self.reg_kwargs = reg_kwargs or {}
            self.reg_fn = reg_fn
            self.reg_coeff = reg_coeff
        else:
            self.reg_kwargs = {}
            self.reg_fn = None
            self.reg_coeff = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.regularized and self.reg_fn:
            reg_term = self.reg_fn(model, **self.reg_kwargs)
            loss = loss+self.reg_coeff * reg_term
        
        return (loss, outputs) if return_outputs else loss
