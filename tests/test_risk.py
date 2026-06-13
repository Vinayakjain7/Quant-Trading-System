from quant import risk

RISK = dict(atr_stop_mult=2.0, risk_per_trade=0.02, max_allocation=0.2,
            max_positions=5, max_drawdown=-0.1)


def test_risk_limit_caps_loss():
    # Risk 2% of 100k = 2000. ATR=5, stop dist=10 -> qty ~ 200, risk ~ 2000.
    r = risk.position_size(100000, 100, atr=5, risk_cfg=RISK)
    assert r.quantity > 0
    assert r.risk_amount <= 100000 * RISK["risk_per_trade"] + 1e-6


def test_allocation_cap():
    # Cheap stock, tiny ATR -> allocation cap (20% = 20000) should bind.
    r = risk.position_size(100000, 10, atr=0.01, risk_cfg=RISK)
    assert r.notional <= 100000 * RISK["max_allocation"] + 10


def test_zero_on_bad_price():
    r = risk.position_size(100000, 0, atr=1, risk_cfg=RISK)
    assert r.quantity == 0


def test_drawdown_halt():
    assert risk.drawdown_halt(-11000, 100000, RISK) is True
    assert risk.drawdown_halt(-5000, 100000, RISK) is False


def test_max_positions():
    assert risk.can_open_new(4, RISK) is True
    assert risk.can_open_new(5, RISK) is False
