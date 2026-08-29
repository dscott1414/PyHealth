# Your Name(s) : David Scott and Reed Rogers
# Your NetId(s): davidrs5, reedwr2
# The paper title: EHR Safari: Data Is Contextual
# The paper link: https://proceedings.mlr.press/v182/boag22a/boag22a.pdf

### How these projects extend *EHR Safari*

# | Paper Section | PyHealth Project | What we measure / validate |
# |---------------|-----------------|----------------------------|
# | 5.2 Inconsistent Timestamps |
#       Build an automatic *flag* so researchers can exclude (or further inspect) 
#       problematic stays. |
# | 5.3 Lab Diurnal Variation | 
#       Quantifies *how much* hour-of-day alone predicts labs — underscoring 
#       the cautionary message about clinical context. |
# | 5.4 Duplicate Notes |
#       Provides a reproducible pipeline to de-duplicate provider notes, preventing 
#       inflated token counts in NLP studies. |
# All three scripts share the PyHealth idioms illustrated in the six reference pipelines 
# (dataset → task fn → split → model → trainer), but are tailored to the specific 
# data-quality caveats highlighted by Boag et al.

# A description of the examples you are implementing:
#   For quick experimentation we point to the **synthetic MIMIC-III subset** hosted by PyHealth; 
#   simply replace the `root=…` argument with your own PostgreSQL/CSV path 
#   if you have the full database.
#   
# Timestamp-Integrity Classifier  
#   Paper § 5.2 — “Inconsistent Timestamps”*
#   Goal  
#      Given an ICU stay’s structured features (diagnoses, procedures, basic demographics), 
#      predict whether its four key timestamps are chronologically consistent**:
#        admittime ≤ intime ≤ outtime ≤ dischtime
#      This can be used as a *QA filter* before downstream modelling or to surface data-entry 
#      patterns to hospital administrators.
def time_integrity_classifier():
    """ Timestamp-Integrity Classifier

        Docs:
            - patients: https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/
            - procedures: https://mimic.mit.edu/docs/iv/modules/hosp/procedures/
            - diagnoses: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses/

        Args:
            None

        Returns:
            An accuracy score having some relationship to the corresponding finding in the paper.
        """

    # --- PyHealth + torch imports ------------------------------------------------
    import torch, torch.nn as nn
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.models import RNN
    from pyhealth.datasets.splitter import split_by_patient
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer

    # ---------- 1. load dataset (synthetic sample) ------------------------------
    mimic_ds = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD"], # "ICUSTAYS",  not implemented
    )

    # ---------- 2. define customised task function ------------------------------
    def timestamp_integrity_fn(patient):
        """
        Returns one binary sample per visit:
        label=1 if timestamps are in correct chronological order
        label=0 otherwise
        Feature keys: diagnoses, procedures
        """
        samples = []
        for visit in patient:
            ok = int(
                visit.encounter_time <= visit.discharge_time
            )
            if len(visit.get_code_list(table="DIAGNOSES_ICD"))>0 and \
                len(visit.get_code_list(table="PROCEDURES_ICD"))>0:
                samples.append(
                    dict(
                        patient_id=patient.patient_id,
                        visit_id=visit.visit_id,
                        diagnoses=visit.get_code_list(table="DIAGNOSES_ICD"),
                        procedures=visit.get_code_list(table="PROCEDURES_ICD"),
                        label=ok,
                    )
                )
        return samples

    task_ds = mimic_ds.set_task(timestamp_integrity_fn)
    print(task_ds.stat())     # quick sanity check

    # ---------- 3. dataloaders ---------------------------------------------------
    train_ds, val_ds, test_ds = split_by_patient(task_ds, [0.7, 0.1, 0.2])
    train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
    val_loader   = get_dataloader(val_ds, batch_size=64)
    test_loader  = get_dataloader(test_ds, batch_size=64)

    # ---------- 4. model ---------------------------------------------------------
    model = RNN(
        dataset=task_ds,
        feature_keys=["diagnoses", "procedures"],
        label_key="label",
        mode="multiclass",
        embedding_dim=64,
        hidden_dim=128,
        rnn_type="GRU",
    )

    # ---------- 5. training ------------------------------------------------------
    trainer = Trainer(
        model=model,
        
        metrics=["accuracy", "roc_auc_macro_ovo"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainer.train(epochs=5, train_dataloader=train_loader,
        val_dataloader=val_loader)

    # ---------- 6. evaluation ----------------------------------------------------
    print(trainer.evaluate(test_loader))

def list_nested_levels(l):
    """Gets all the different nested levels of a list.

    Args:
        l: the list to be checked.

    Returns:
        All the different nested levels of the list.

    Examples:
        >>> list_nested_levels([])
        (1,)
        >>> list_nested_levels([1, 2, 3])
        (1,)
        >>> list_nested_levels([[]])
        (2,)
        >>> list_nested_levels([[1, 2, 3], [4, 5, 6]])
        (2,)
        >>> list_nested_levels([1, [2, 3], 4])
        (1, 2)
        >>> list_nested_levels([[1, [2, 3], 4]])
        (2, 3)
    """
    if not isinstance(l, list):
        return tuple([0])
    if not l:
        return tuple([1])
    levels = []
    for i in l:
        levels.extend(list_nested_levels(i))
    levels = [i + 1 for i in levels]
    return tuple(set(levels))

def validate(samples):
    """Helper function which validates the samples.

    Will be called in `self.__init__()`.

    Returns:
        input_info: Dict, a dict whose keys are the same as the keys in the
            samples, and values are the corresponding input information:
            - "type": the element type of each key attribute, one of float,
                int, str.
            - "dim": the list dimension of each key attribute, one of 0, 1, 2, 3.
            - "len": the length of the vector, only valid for vector-based
                attributes.
    """
    """ 1. Check if all samples are of type dict. """
    assert all(
        [isinstance(s, dict) for s in samples],
    ), "Each sample should be a dict"
    keys = samples[0].keys()
    print(f"keys={keys}")
    """ 2. Check if all samples have the same keys. """
    assert all(
        [set(s.keys()) == set(keys) for s in samples]
    ), "All samples should have the same keys"

    """ 3. Check if "patient_id" and "visit_id" are in the keys."""
    assert "patient_id" in keys, "patient_id should be in the keys"
    assert "visit_id" in keys, "visit_id should be in the keys"

    """
    4. For each key, check if it is either:
        - a single value
        - a single vector
        - a list of codes
        - a list of vectors
        - a list of list of codes
        - a list of list of vectors
    Note that a value is either float, int, or str; a vector is a list of float 
    or int; and a code is str.
    """
    # record input information for each key
    input_info = {}
    for key in keys:
        """
        4.1. Check nested list level: all samples should either all be
        - a single value (level=0)
        - a single vector (level=1)
        - a list of codes (level=1)
        - a list of vectors (level=2)
        - a list of list of codes (level=2)
        - a list of list of vectors (level=3)
        """
        levels = set([list_nested_levels(s[key]) for s in samples])
        assert (
            len(levels) == 1 and len(list(levels)[0]) == 1
        ), f"Key {key} has mixed nested list levels across samples"
        level = levels.pop()[0]
        assert level in [
            0,
            1,
            2,
            3,
        ], f"Key {key} has unsupported nested list level across samples"
        print(levels,f"level={level}")
        # flatten the list
        if level == 0:
            flattened_values = [s[key] for s in samples]
        elif level == 1:
            flattened_values = [i for s in samples for i in s[key]]
        elif level == 2:
            flattened_values = [j for s in samples for i in s[key] for j in i]
        else:
            flattened_values = [
                k for s in samples for i in s[key] for j in i for k in j
            ]
        print(f"flattened_values={flattened_values}")

        """
        4.2. Check type: the basic type of each element should be float, 
        int, or str.
        """
        types = set([type(v) for v in flattened_values])
        assert (
            types == set([str]) or len(types.difference(set([int, float]))) == 0
        ), f"Key {key} has mixed or unsupported types ({types}) across samples"
        type_ = types.pop()
        """
        4.3. Combined level and type check.
        """
        if level == 0:
            # a single value
            input_info[key] = {"type": type_, "dim": 0}
        elif level == 1:
            # a single vector or a list of codes
            if type_ in [float, int]:
                # a single vector
                lens = set([len(s[key]) for s in samples])
                assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                input_info[key] = {"type": type_, "dim": 1, "len": lens.pop()}
            else:
                # a list of codes
                # note that dim is different from level here
                input_info[key] = {"type": type_, "dim": 2}
        elif level == 2:
            # a list of vectors or a list of list of codes
            if type_ in [float, int]:
                lens = set([len(i) for s in samples for i in s[key]])
                assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                input_info[key] = {"type": type_, "dim": 2, "len": lens.pop()}
            else:
                # a list of list of codes
                # note that dim is different from level here
                input_info[key] = {"type": type_, "dim": 3}
        else:
            # a list of list of vectors
            assert type_ in [
                float,
                int,
            ], f"Key {key} has unsupported type across samples"
            lens = set([len(j) for s in samples for i in s[key] for j in i])
            assert len(lens) == 1, f"Key {key} has vectors of different lengths"
            input_info[key] = {"type": type_, "dim": 3, "len": lens.pop()}
        print("passed")

    
##   
# Paper § 5.3 — “Lab Values Vary by Time of Day”*
def diurnal_lab_abnormality_predictor():
    """ Lab Values Vary by Time of Day
        Goal  
        Create a time-aware classifier that attempts to predict labs and diagnoses by the time.
        This illustrates how ignoring timestamp context can hide strong time-of-day priors.

        Docs:
            - patients: https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/
            - procedures: https://mimic.mit.edu/docs/iv/modules/hosp/procedures/
            - diagnoses: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses/

        Args:
            None

        Returns:
            An accuracy score having some relationship to the corresponding finding in the paper.
        """

    import torch, torch.nn as nn
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.models import deepr
    from pyhealth.datasets.splitter import split_by_patient
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer
    import numpy as np
    import datetime
    print("Loading dataset...")
    # ---- 1. dataset -------------------------------------------------------------
    mimic_ds = MIMIC3Dataset(
        root="/users/davidscott/physionet.org/files/mimiciii/1.4/",
        tables=["LABEVENTS", "DIAGNOSES_ICD"],
    )

    # ---- 2. task function -------------------------------------------------------
    def wbc_diurnal_fn(patient):
        samples = []
        for visit in patient:
            slist = []
            for s in visit.get_code_list(table="DIAGNOSES_ICD") + visit.get_code_list(table="PROCEDURES_ICD") + visit.get_code_list(table="LABEVENTS"):
                slist += str(s)
            if len(slist)>0:
                samples.append(
                        dict(
                            patient_id=int(patient.patient_id),
                            visit_id=int(visit.visit_id),
                            samples=slist,
                            label=visit.encounter_time.hour,
                        )
                    )
        if len(samples)==0:
            samples.append(
                    dict(
                        patient_id=0,
                        visit_id=0,
                        samples=["0"],
                        label=0,
                    )
                )
        return samples

    print("Processing dataset...")
    task_ds = mimic_ds.set_task(wbc_diurnal_fn)
    print(task_ds.stat())

    # ---- 3. split & loader ------------------------------------------------------
    train_ds, val_ds, test_ds = split_by_patient(task_ds, [0.7,0.1,0.2])
    train_loader = get_dataloader(train_ds, batch_size=256, shuffle=True)
    val_loader   = get_dataloader(val_ds, batch_size=256)
    test_loader  = get_dataloader(test_ds, batch_size=256)

    # ---- 4. model (Wide + Deep) -------------------------------------------------
    model = RNN(
        dataset=task_ds,
        feature_keys=["samples"],
        label_key="label",
        mode="multiclass",
        embedding_dim=64,
        hidden_dim=128,
        rnn_type="GRU",
    )

    # ---- 5. train ---------------------------------------------------------------
    trainer = Trainer(
        model=model,
        #
        metrics=["accuracy"],
    )
    trainer.train(epochs=3,train_dataloader=train_loader,val_dataloader=val_loader)
    print(trainer.evaluate(test_loader))

# note: this example will run when NOTEEVENTS is supported in the MIMIC3Dataset class.
# Paper § 5.4 — “Multiple Copies of Provider Notes”*
def duplicate_note_detector():
    """ Lab Values Vary by Time of Day
        Goal  
        Identify draft duplicates of a provider note written within the same `charttime` group 
        so that downstream NLP pipelines are not inflated by repeated text.
        We treat this as a pairwise-similarity classification** task: for every 
        `(note_A, note_B)` with the same `(subject_id, hadm_id, charttime)` 
        decide whether they are semantically identical (`label=1`) or not (`label=0`). 
        A `MiniLM` text-encoder (through *Sentence-Transformers*) feeds a simple MLP.

        Docs:
            - patients: https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/
            - procedures: https://mimic.mit.edu/docs/iv/modules/hosp/procedures/
            - diagnoses: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses/

        Args:
            None

        Returns:
            An accuracy score having some relationship to the corresponding finding in the paper.
        """

    import torch, torch.nn as nn
    from pyhealth.datasets import MIMIC3Dataset
    from itertools import combinations
    from sklearn.metrics.pairwise import cosine_similarity
    from pyhealth.datasets.splitter import split_by_patient
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer
    import numpy as np
    print("Loading dataset...")
    # ---- 1. dataset -------------------------------------------------------------
    mimic_ds = MIMIC3Dataset(
        root="/users/davidscott/physionet.org/files/mimiciii/1.4/",
        tables=["NOTEEVENTS"],
    )

    # ---- 2. task: build note pairs ---------------------------------------------
    def duplicate_note_fn(patient):
        samples=[]
        # group notes by (hadm_id, charttime)
        visits = patient.get_table("NOTEEVENTS")
        group={}
        for note in visits:
            key=(note["hadm_id"], note["charttime"])
            group.setdefault(key, []).append(note)
        for g in group.values():
            for a,b in combinations(g,2):
                label=int(a["text"].strip()==b["text"].strip())
                samples.append(
                    dict(
                        patient_id=patient.patient_id,
                        visit_id  =a["row_id"],          # arbitrary
                        text_a=a["text"][:1000],         # cut long notes in demo
                        text_b=b["text"][:1000],
                        label=label,
                    )
                )
        return samples

    task_ds = mimic_ds.set_task(duplicate_note_fn)
    print(task_ds.stat())

    # ---- 3. sentence-transformer encoder (frozen) ------------------------------
    sbert = SentenceTransformer("all-MiniLM-L6-v2")   # 384-d embeddings
    sbert.eval()

    def embed(batch):
        with torch.no_grad():
            ea = torch.tensor(sbert.encode(batch["text_a"], convert_to_numpy=True))
            eb = torch.tensor(sbert.encode(batch["text_b"], convert_to_numpy=True))
        return torch.cat([ea, eb, torch.abs(ea-eb)], dim=1)   # 3*384 features

    # ---- 4. simple MLP classifier ----------------------------------------------
    class DuplicateMLP(nn.Module):
        def __init__(self, in_dim=3*384):
            super().__init__()
            self.net=nn.Sequential(
                nn.Linear(in_dim,256),
                nn.ReLU(),
                nn.Linear(256,1)
            )
        def forward(self,x): return self.net(x).squeeze(1)

    model = DuplicateMLP()

    # ---- 5. data loaders --------------------------------------------------------
    train_ds, val_ds, test_ds = split_by_patient(task_ds, [0.7,0.1,0.2])

    def collate_fn(batch):
        feats=embed({k:[d[k] for d in batch] for k in ("text_a","text_b")})
        labels=torch.tensor([d["label"] for d in batch], dtype=torch.float32)
        return feats, labels

    train_loader=get_dataloader(train_ds, batch_size=128, collate_fn=collate_fn, shuffle=True)
    val_loader  =get_dataloader(val_ds , batch_size=128, collate_fn=collate_fn)
    test_loader =get_dataloader(test_ds, batch_size=128, collate_fn=collate_fn)

    # ---- 6. training loop (simplified) -----------------------------------------
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn=nn.BCEWithLogitsLoss()

    def run_epoch(loader,train):
        tot,correct,loss_sum=0,0,0.0
        model.train(train)
        for x,y in loader:
            if train: optimizer.zero_grad()
            out=model(x)
            loss=loss_fn(out,y)
            if train:
                loss.backward(); optimizer.step()
            preds=(torch.sigmoid(out)>0.5).long()
            correct+= (preds==y.long()).sum().item()
            tot+=len(y); loss_sum+=loss.item()*len(y)
        return loss_sum/tot, correct/tot

    for epoch in range(3):
        tr_loss,tr_acc=run_epoch(train_loader,True)
        vl_loss,vl_acc=run_epoch(val_loader,False)
        print(f"epoch{epoch}: train_acc={tr_acc:.3f} val_acc={vl_acc:.3f}")

    # ---- 7. final test ----------------------------------------------------------
    test_loss,test_acc=run_epoch(test_loader,False)
    print("test accuracy:",test_acc)

