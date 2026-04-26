def calculate_risk(predicted_price, base_price):
    change = (predicted_price - base_price) / base_price

    if change > 0.2:
        return "Low Risk"
    elif change > 0:
        return "Medium Risk"
    else:
        return "High Risk"


def simulate_price(base_price, interest_rate, demand_factor=1.0):
    growth_rate = 0.08

    # Interest reduces price
    interest_impact = interest_rate * 0.02

    # Demand increases price
    demand_impact = demand_factor * 0.05

    return base_price * (1 + growth_rate - interest_impact + demand_impact)