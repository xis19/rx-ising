use rx_ising::ising::Ising;
use rx_ising::montecarlo::MonteCarlo;

fn main() {
    let mut mc =
        MonteCarlo::<Ising<2, false>>::new(0.01, 0.5, Ising::<2, false>::new(0.0, 0.1, [20, 20]));
    let rep = 100000;

    for _ in 0..rep {
        mc.next_sample();
        let order_parameter = mc.get_sample().get_order_parameter();
        mc.set_ensemble_average("OrderParameter", order_parameter);
    }

    let mc_stats = mc.get_sample_counter();
    println!("Acceptance rate = {}", mc_stats.get_acceptance_rate());

    let average_energy = mc.get_ensemble_average("Energy").unwrap();
    println!("Average energy = {}", average_energy.get_average());
    println!(
        "Energy standard derivation= {}",
        average_energy.get_standard_derivation()
    );

    let order_parameter = mc.get_ensemble_average("OrderParameter").unwrap();
    println!("Average order parameter = {}", order_parameter.get_average());
    println!(
        "Energy standard derivation = {}",
        order_parameter.get_standard_derivation()
    );
}
