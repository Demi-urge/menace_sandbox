import menace.offer_testing_bot as ot


def test_variation_generation(tmp_path):
    db = ot.OfferDB(tmp_path / "o.db")
    bot = ot.OfferTestingBot(db)
    ids = bot.generate_variations("prod", 10.0)
    assert len(ids) == 6
    vars = db.list_variants()
    assert len(vars) == 6


def test_promote_winners(tmp_path):
    db = ot.OfferDB(tmp_path / "o.db")
    bot = ot.OfferTestingBot(db)
    ids = bot.generate_variations("prod", 10.0)
    # log interactions: first variant has better conversion
    db.log_interaction(ot.OfferInteraction(ids[0], True, 10.0, True, False, 1.0))
    db.log_interaction(ot.OfferInteraction(ids[1], False, 0.0, False, False, 2.0))
    bot.promote_winners(top_n=1)
    actives = [v.id for v in db.list_variants(active=True)]
    assert actives == [ids[0]]
