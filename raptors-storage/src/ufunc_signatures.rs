//! Numeric ufunc loop metadata extracted from NumPy 2.5.3 on macOS arm64.
//! Non-numeric datetime, object, and string loops are intentionally omitted.
//! The table is data only; Raptors executes the corresponding Rust scalar kernels.

pub fn signatures(name: &str) -> Vec<String> {
    let signatures: &[&str] = match name {
        "absolute" => &[
            "?->?", "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q",
            "e->e", "f->f", "d->d", "g->g", "F->f", "D->d", "G->g",
        ],
        "add" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g", "FF->F", "DD->D", "GG->G",
        ],
        "arccos" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "arccosh" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "arcsin" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "arcsinh" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "arctan" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "arctan2" => &["ee->e", "ff->f", "dd->d", "gg->g"],
        "arctanh" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "bitwise_and" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q",
        ],
        "bitwise_count" => &[
            "b->B", "B->B", "h->B", "H->B", "i->B", "I->B", "l->B", "L->B", "q->B", "Q->B",
        ],
        "bitwise_or" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q",
        ],
        "bitwise_xor" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q",
        ],
        "cbrt" => &["e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g"],
        "ceil" => &[
            "?->?", "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q",
            "e->e", "f->f", "d->d", "f->f", "d->d", "g->g",
        ],
        "conjugate" => &[
            "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q", "e->e",
            "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "copysign" => &["ee->e", "ff->f", "dd->d", "gg->g"],
        "cos" => &["e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G"],
        "cosh" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "deg2rad" => &["e->e", "f->f", "d->d", "g->g"],
        "degrees" => &["e->e", "f->f", "d->d", "g->g"],
        "divide" => &[
            "ee->e", "ff->f", "dd->d", "gg->g", "FF->F", "DD->D", "GG->G",
        ],
        "divmod" => &[
            "bb->bb", "BB->BB", "hh->hh", "HH->HH", "ii->ii", "II->II", "ll->ll", "LL->LL",
            "qq->qq", "QQ->QQ", "ee->ee", "ff->ff", "dd->dd", "gg->gg",
        ],
        "equal" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "qQ->?", "Qq->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?",
            "DD->?", "GG->?",
        ],
        "exp" => &[
            "e->e", "f->f", "d->d", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "exp2" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "expm1" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "fabs" => &["e->e", "f->f", "d->d", "g->g"],
        "float_power" => &["dd->d", "gg->g", "DD->D", "GG->G"],
        "floor" => &[
            "?->?", "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q",
            "e->e", "f->f", "d->d", "f->f", "d->d", "g->g",
        ],
        "floor_divide" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g",
        ],
        "fmax" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g", "FF->F", "DD->D", "GG->G",
        ],
        "fmin" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g", "FF->F", "DD->D", "GG->G",
        ],
        "fmod" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g",
        ],
        "frexp" => &["e->ei", "f->fi", "d->di", "g->gi"],
        "gcd" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q",
        ],
        "greater" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "qQ->?", "Qq->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?",
            "DD->?", "GG->?",
        ],
        "greater_equal" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "qQ->?", "Qq->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?",
            "DD->?", "GG->?",
        ],
        "heaviside" => &["ee->e", "ff->f", "dd->d", "gg->g"],
        "hypot" => &["ee->e", "ff->f", "dd->d", "gg->g"],
        "invert" => &[
            "?->?", "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q",
        ],
        "isfinite" => &[
            "?->?", "b->?", "B->?", "h->?", "H->?", "i->?", "I->?", "l->?", "L->?", "q->?", "Q->?",
            "e->?", "f->?", "d->?", "g->?", "F->?", "D->?", "G->?",
        ],
        "isinf" => &[
            "?->?", "b->?", "B->?", "h->?", "H->?", "i->?", "I->?", "l->?", "L->?", "q->?", "Q->?",
            "e->?", "f->?", "d->?", "g->?", "F->?", "D->?", "G->?",
        ],
        "isnan" => &[
            "?->?", "b->?", "B->?", "h->?", "H->?", "i->?", "I->?", "l->?", "L->?", "q->?", "Q->?",
            "e->?", "f->?", "d->?", "g->?", "F->?", "D->?", "G->?",
        ],
        "lcm" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q",
        ],
        "ldexp" => &[
            "ei->e", "fi->f", "el->e", "fl->f", "di->d", "dl->d", "gi->g", "gl->g",
        ],
        "left_shift" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q",
        ],
        "less" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "qQ->?", "Qq->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?",
            "DD->?", "GG->?",
        ],
        "less_equal" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "qQ->?", "Qq->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?",
            "DD->?", "GG->?",
        ],
        "log" => &[
            "e->e", "f->f", "d->d", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "log10" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "log1p" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "log2" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "logaddexp" => &["ee->e", "ff->f", "dd->d", "gg->g"],
        "logaddexp2" => &["ee->e", "ff->f", "dd->d", "gg->g"],
        "logical_and" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?", "DD->?", "GG->?",
        ],
        "logical_not" => &[
            "?->?", "b->?", "B->?", "h->?", "H->?", "i->?", "I->?", "l->?", "L->?", "q->?", "Q->?",
            "e->?", "f->?", "d->?", "g->?", "F->?", "D->?", "G->?",
        ],
        "logical_or" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?", "DD->?", "GG->?",
        ],
        "logical_xor" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?", "DD->?", "GG->?",
        ],
        "maximum" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g", "FF->F", "DD->D", "GG->G",
        ],
        "minimum" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g", "FF->F", "DD->D", "GG->G",
        ],
        "modf" => &["e->ee", "f->ff", "d->dd", "g->gg"],
        "multiply" => &[
            "??->?", "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L",
            "qq->q", "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g", "FF->F", "DD->D", "GG->G",
        ],
        "negative" => &[
            "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q", "e->e",
            "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "nextafter" => &["ee->e", "ff->f", "dd->d", "gg->g"],
        "not_equal" => &[
            "??->?", "bb->?", "BB->?", "hh->?", "HH->?", "ii->?", "II->?", "ll->?", "LL->?",
            "qq->?", "QQ->?", "qQ->?", "Qq->?", "ee->?", "ff->?", "dd->?", "gg->?", "FF->?",
            "DD->?", "GG->?",
        ],
        "positive" => &[
            "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q", "e->e",
            "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "power" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q", "ee->e", "ff->f", "dd->d", "ee->e", "ff->f", "dd->d", "gg->g", "FF->F",
            "DD->D", "GG->G",
        ],
        "rad2deg" => &["e->e", "f->f", "d->d", "g->g"],
        "radians" => &["e->e", "f->f", "d->d", "g->g"],
        "reciprocal" => &[
            "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q", "e->e",
            "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "remainder" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g",
        ],
        "right_shift" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q",
        ],
        "rint" => &[
            "e->e", "f->f", "d->d", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "sign" => &[
            "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q", "e->e",
            "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "signbit" => &["e->?", "f->?", "d->?", "g->?"],
        "sin" => &["e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G"],
        "sinh" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "spacing" => &["e->e", "f->f", "d->d", "g->g"],
        "sqrt" => &[
            "e->e", "f->f", "d->d", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "square" => &[
            "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q", "e->e",
            "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "subtract" => &[
            "bb->b", "BB->B", "hh->h", "HH->H", "ii->i", "II->I", "ll->l", "LL->L", "qq->q",
            "QQ->Q", "ee->e", "ff->f", "dd->d", "gg->g", "FF->F", "DD->D", "GG->G",
        ],
        "tan" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "tanh" => &[
            "e->e", "f->f", "d->d", "e->e", "f->f", "d->d", "g->g", "F->F", "D->D", "G->G",
        ],
        "trunc" => &[
            "?->?", "b->b", "B->B", "h->h", "H->H", "i->i", "I->I", "l->l", "L->L", "q->q", "Q->Q",
            "e->e", "f->f", "d->d", "f->f", "d->d", "g->g",
        ],
        _ => &[],
    };
    signatures
        .iter()
        .map(|signature| target_signature(signature))
        .collect()
}

fn target_signature(signature: &str) -> String {
    if !cfg!(target_os = "windows") {
        return signature.to_owned();
    }
    windows_signature(signature)
}

fn windows_signature(signature: &str) -> String {
    signature
        .chars()
        .map(|code| match code {
            'l' => 'q',
            'L' => 'Q',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::windows_signature;

    #[test]
    fn windows_typecodes_preserve_c_int_and_map_c_long_to_longlong() {
        assert_eq!(windows_signature("ii->iII->I"), "ii->iII->I");
        assert_eq!(
            windows_signature("ll->lLL->Lqq->qQQ->Q"),
            "qq->qQQ->Qqq->qQQ->Q"
        );
    }
}
