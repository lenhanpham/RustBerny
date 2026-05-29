//! Element property database.
//!
//! Provides atomic properties (covalent radius, van der Waals radius, mass)
//! for all elements. Ghost atoms (symbol "X", "BQ", "GHOST") return zero
//! for all properties.

use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Properties for a single element.
#[derive(Debug, Clone)]
pub struct ElementData {
    /// Atomic number.
    pub number: u32,
    /// Element name.
    pub name: &'static str,
    /// Element symbol.
    pub symbol: &'static str,
    /// Covalent radius in angstrom.
    pub covalent_radius: f64,
    /// Atomic mass in amu.
    pub mass: f64,
    /// Van der Waals radius in angstrom.
    pub vdw_radius: f64,
}

/// Ghost atom aliases (case-insensitive, after stripping leading dashes).
const GHOST_ALIASES: &[&str] = &["ghost", "x", "bq"];

/// Returns `true` if `symbol` represents a ghost/basis-set center.
///
/// Ghost atoms have zero covalent radius, zero van der Waals radius,
/// and zero mass. They are never bonded and do not contribute to
/// angles or dihedrals.
pub fn is_ghost(symbol: &str) -> bool {
    let stripped = symbol.strip_prefix('-').unwrap_or(symbol);
    GHOST_ALIASES.contains(&stripped.to_lowercase().as_str())
}

/// Returns zero-valued properties for ghost atoms.
fn ghost_row(property: &str) -> f64 {
    match property {
        "number" | "covalent_radius" | "vdw_radius" | "mass" => 0.0,
        _ => 0.0,
    }
}

/// Hardcoded element data for all 92 elements (H through U).
const ELEMENTS: &[ElementData] = &[
    ElementData { number: 1, name: "hydrogen", symbol: "H", covalent_radius: 0.38, mass: 1.0079, vdw_radius: 1.6404493538 },
    ElementData { number: 2, name: "helium", symbol: "He", covalent_radius: 0.32, mass: 4.0026, vdw_radius: 1.4023196089 },
    ElementData { number: 3, name: "lithium", symbol: "Li", covalent_radius: 1.34, mass: 6.941, vdw_radius: 2.2013771973 },
    ElementData { number: 4, name: "beryllium", symbol: "Be", covalent_radius: 0.9, mass: 9.0122, vdw_radius: 2.2066689695 },
    ElementData { number: 5, name: "boron", symbol: "B", covalent_radius: 0.82, mass: 10.811, vdw_radius: 2.0584993504 },
    ElementData { number: 6, name: "carbon", symbol: "C", covalent_radius: 0.77, mass: 12.0107, vdw_radius: 1.8997461871 },
    ElementData { number: 7, name: "nitrogen", symbol: "N", covalent_radius: 0.75, mass: 14.0067, vdw_radius: 1.7674518844 },
    ElementData { number: 8, name: "oxygen", symbol: "O", covalent_radius: 0.73, mass: 15.9994, vdw_radius: 1.6880753028 },
    ElementData { number: 9, name: "fluorine", symbol: "F", covalent_radius: 0.71, mass: 18.9984, vdw_radius: 1.6086987211 },
    ElementData { number: 10, name: "neon", symbol: "Ne", covalent_radius: 0.69, mass: 20.1797, vdw_radius: 1.5399056837 },
    ElementData { number: 11, name: "sodium", symbol: "Na", covalent_radius: 1.54, mass: 22.9897, vdw_radius: 1.9738309967 },
    ElementData { number: 12, name: "magnesium", symbol: "Mg", covalent_radius: 1.3, mass: 24.305, vdw_radius: 2.2595866905 },
    ElementData { number: 13, name: "aluminium", symbol: "Al", covalent_radius: 1.18, mass: 26.9815, vdw_radius: 2.2913373232 },
    ElementData { number: 14, name: "silicon", symbol: "Si", covalent_radius: 1.11, mass: 28.0855, vdw_radius: 2.2225442858 },
    ElementData { number: 15, name: "phosphorus", symbol: "P", covalent_radius: 1.06, mass: 30.9738, vdw_radius: 2.1220006157 },
    ElementData { number: 16, name: "sulfur", symbol: "S", covalent_radius: 1.02, mass: 32.065, vdw_radius: 2.0426240341 },
    ElementData { number: 17, name: "chlorine", symbol: "Cl", covalent_radius: 0.99, mass: 35.453, vdw_radius: 1.9632474524 },
    ElementData { number: 18, name: "argon", symbol: "Ar", covalent_radius: 0.97, mass: 39.948, vdw_radius: 1.8785790987 },
    ElementData { number: 19, name: "potassium", symbol: "K", covalent_radius: 1.96, mass: 39.0983, vdw_radius: 1.9632474524 },
    ElementData { number: 20, name: "calcium", symbol: "Ca", covalent_radius: 1.74, mass: 40.078, vdw_radius: 2.4606740307 },
    ElementData { number: 21, name: "scandium", symbol: "Sc", covalent_radius: 1.44, mass: 44.9559, vdw_radius: 2.428923398 },
    ElementData { number: 22, name: "titanium", symbol: "Ti", covalent_radius: 1.36, mass: 47.867, vdw_radius: 2.3865892212 },
    ElementData { number: 23, name: "vanadium", symbol: "V", covalent_radius: 1.25, mass: 50.9415, vdw_radius: 2.3495468164 },
    ElementData { number: 24, name: "chromium", symbol: "Cr", covalent_radius: 1.27, mass: 51.9961, vdw_radius: 2.1114170715 },
    ElementData { number: 25, name: "manganese", symbol: "Mn", covalent_radius: 1.39, mass: 54.938, vdw_radius: 2.1008335273 },
    ElementData { number: 26, name: "iron", symbol: "Fe", covalent_radius: 1.25, mass: 55.845, vdw_radius: 2.2384196021 },
    ElementData { number: 27, name: "cobalt", symbol: "Co", covalent_radius: 1.26, mass: 58.9332, vdw_radius: 2.2119607416 },
    ElementData { number: 28, name: "nickel", symbol: "Ni", covalent_radius: 1.21, mass: 58.6934, vdw_radius: 2.0214569456 },
    ElementData { number: 29, name: "copper", symbol: "Cu", covalent_radius: 1.38, mass: 63.546, vdw_radius: 1.989706313 },
    ElementData { number: 30, name: "zinc", symbol: "Zn", covalent_radius: 1.31, mass: 65.39, vdw_radius: 2.1272923878 },
    ElementData { number: 31, name: "gallium", symbol: "Ga", covalent_radius: 1.26, mass: 69.723, vdw_radius: 2.2172525137 },
    ElementData { number: 32, name: "germanium", symbol: "Ge", covalent_radius: 1.22, mass: 72.64, vdw_radius: 2.2225442858 },
    ElementData { number: 33, name: "arsenic", symbol: "As", covalent_radius: 1.19, mass: 74.9216, vdw_radius: 2.1749183368 },
    ElementData { number: 34, name: "selenium", symbol: "Se", covalent_radius: 1.16, mass: 78.96, vdw_radius: 2.137875932 },
    ElementData { number: 35, name: "bromine", symbol: "Br", covalent_radius: 1.14, mass: 79.904, vdw_radius: 2.0796664388 },
    ElementData { number: 36, name: "krypton", symbol: "Kr", covalent_radius: 1.1, mass: 83.8, vdw_radius: 2.0214569456 },
    ElementData { number: 37, name: "rubidium", symbol: "Rb", covalent_radius: 2.11, mass: 85.4678, vdw_radius: 1.9685392245 },
    ElementData { number: 38, name: "strontium", symbol: "Sr", covalent_radius: 1.92, mass: 87.62, vdw_radius: 2.4024645375 },
    ElementData { number: 39, name: "yttrium", symbol: "Y", covalent_radius: 1.62, mass: 88.9059, vdw_radius: 2.5480411882 },
    ElementData { number: 40, name: "zirconium", symbol: "Zr", covalent_radius: 1.48, mass: 91.224, vdw_radius: 2.3971727654 },
    ElementData { number: 41, name: "niobium", symbol: "Nb", covalent_radius: 1.37, mass: 92.9064, vdw_radius: 2.241859254 },
    ElementData { number: 42, name: "molybdenum", symbol: "Mo", covalent_radius: 1.45, mass: 95.94, vdw_radius: 2.1690973875 },
    ElementData { number: 43, name: "technetium", symbol: "Tc", covalent_radius: 1.56, mass: 98.0, vdw_radius: 2.1569263116 },
    ElementData { number: 44, name: "ruthenium", symbol: "Ru", covalent_radius: 1.26, mass: 101.07, vdw_radius: 2.1142217107 },
    ElementData { number: 45, name: "rhodium", symbol: "Rh", covalent_radius: 1.35, mass: 102.9055, vdw_radius: 2.0902499831 },
    ElementData { number: 46, name: "palladium", symbol: "Pd", covalent_radius: 1.31, mass: 106.42, vdw_radius: 1.9367885919 },
    ElementData { number: 47, name: "silver", symbol: "Ag", covalent_radius: 1.53, mass: 107.8682, vdw_radius: 2.0214569456 },
    ElementData { number: 48, name: "cadmium", symbol: "Cd", covalent_radius: 1.48, mass: 112.411, vdw_radius: 2.1114170715 },
    ElementData { number: 49, name: "indium", symbol: "In", covalent_radius: 1.44, mass: 114.818, vdw_radius: 2.239467373 },
    ElementData { number: 50, name: "tin", symbol: "Sn", covalent_radius: 1.41, mass: 118.71, vdw_radius: 2.2770495385 },
    ElementData { number: 51, name: "antimony", symbol: "Sb", covalent_radius: 1.38, mass: 121.76, vdw_radius: 2.2627617538 },
    ElementData { number: 52, name: "tellurium", symbol: "Te", covalent_radius: 1.35, mass: 127.6, vdw_radius: 2.23312783 },
    ElementData { number: 53, name: "iodine", symbol: "I", covalent_radius: 1.33, mass: 126.9045, vdw_radius: 2.2066689695 },
    ElementData { number: 54, name: "xenon", symbol: "Xe", covalent_radius: 1.3, mass: 131.293, vdw_radius: 2.1590430205 },
    ElementData { number: 55, name: "caesium", symbol: "Cs", covalent_radius: 2.25, mass: 132.9055, vdw_radius: 2.0002898572 },
    ElementData { number: 56, name: "barium", symbol: "Ba", covalent_radius: 1.98, mass: 137.327, vdw_radius: 2.524175296 },
    ElementData { number: 57, name: "lanthanum", symbol: "La", covalent_radius: 1.69, mass: 138.9055, vdw_radius: 0.0 },
    ElementData { number: 58, name: "cerium", symbol: "Ce", covalent_radius: 2.04, mass: 140.116, vdw_radius: 0.0 },
    ElementData { number: 59, name: "praseodymium", symbol: "Pr", covalent_radius: 2.03, mass: 140.9077, vdw_radius: 0.0 },
    ElementData { number: 60, name: "neodymium", symbol: "Nd", covalent_radius: 2.01, mass: 144.24, vdw_radius: 0.0 },
    ElementData { number: 61, name: "promethium", symbol: "Pm", covalent_radius: 1.99, mass: 145.0, vdw_radius: 0.0 },
    ElementData { number: 62, name: "samarium", symbol: "Sm", covalent_radius: 1.98, mass: 150.36, vdw_radius: 0.0 },
    ElementData { number: 63, name: "europium", symbol: "Eu", covalent_radius: 1.98, mass: 151.964, vdw_radius: 0.0 },
    ElementData { number: 64, name: "gadolinium", symbol: "Gd", covalent_radius: 1.96, mass: 157.25, vdw_radius: 0.0 },
    ElementData { number: 65, name: "terbium", symbol: "Tb", covalent_radius: 1.94, mass: 158.9253, vdw_radius: 0.0 },
    ElementData { number: 66, name: "dysprosium", symbol: "Dy", covalent_radius: 1.92, mass: 162.5, vdw_radius: 0.0 },
    ElementData { number: 67, name: "holmium", symbol: "Ho", covalent_radius: 1.92, mass: 164.9303, vdw_radius: 0.0 },
    ElementData { number: 68, name: "erbium", symbol: "Er", covalent_radius: 1.89, mass: 167.259, vdw_radius: 0.0 },
    ElementData { number: 69, name: "thulium", symbol: "Tm", covalent_radius: 1.9, mass: 168.9342, vdw_radius: 0.0 },
    ElementData { number: 70, name: "ytterbium", symbol: "Yb", covalent_radius: 1.87, mass: 173.04, vdw_radius: 0.0 },
    ElementData { number: 71, name: "lutetium", symbol: "Lu", covalent_radius: 1.6, mass: 174.967, vdw_radius: 0.0 },
    ElementData { number: 72, name: "hafnium", symbol: "Hf", covalent_radius: 1.5, mass: 178.49, vdw_radius: 2.2278360579 },
    ElementData { number: 73, name: "tantalum", symbol: "Ta", covalent_radius: 1.38, mass: 180.9479, vdw_radius: 2.1960854252 },
    ElementData { number: 74, name: "tungsten", symbol: "W", covalent_radius: 1.46, mass: 183.84, vdw_radius: 2.1590430205 },
    ElementData { number: 75, name: "rhenium", symbol: "Re", covalent_radius: 1.59, mass: 186.207, vdw_radius: 2.1272923878 },
    ElementData { number: 76, name: "osmium", symbol: "Os", covalent_radius: 1.28, mass: 190.23, vdw_radius: 2.0320404899 },
    ElementData { number: 77, name: "iridium", symbol: "Ir", covalent_radius: 1.37, mass: 192.217, vdw_radius: 2.1167088436 },
    ElementData { number: 78, name: "platinum", symbol: "Pt", covalent_radius: 1.28, mass: 195.078, vdw_radius: 2.0743746667 },
    ElementData { number: 79, name: "gold", symbol: "Au", covalent_radius: 1.44, mass: 196.9665, vdw_radius: 2.0426240341 },
    ElementData { number: 80, name: "mercury", symbol: "Hg", covalent_radius: 1.49, mass: 200.59, vdw_radius: 2.1061252994 },
    ElementData { number: 81, name: "thallium", symbol: "Tl", covalent_radius: 1.48, mass: 204.3833, vdw_radius: 2.0690828946 },
    ElementData { number: 82, name: "lead", symbol: "Pb", covalent_radius: 1.47, mass: 207.2, vdw_radius: 2.280753779 },
    ElementData { number: 83, name: "bismuth", symbol: "Bi", covalent_radius: 1.46, mass: 208.9804, vdw_radius: 2.2860455511 },
    ElementData { number: 84, name: "polonium", symbol: "Po", covalent_radius: 1.4, mass: 209.0, vdw_radius: 2.1680390331 },
    ElementData { number: 85, name: "astatine", symbol: "At", covalent_radius: 1.5, mass: 210.0, vdw_radius: 2.1537512484 },
    ElementData { number: 86, name: "radon", symbol: "Rn", covalent_radius: 1.45, mass: 222.0, vdw_radius: 2.2384196021 },
    ElementData { number: 87, name: "francium", symbol: "Fr", covalent_radius: 2.6, mass: 223.0, vdw_radius: 0.0 },
    ElementData { number: 88, name: "radium", symbol: "Ra", covalent_radius: 2.21, mass: 226.0, vdw_radius: 0.0 },
    ElementData { number: 89, name: "actinium", symbol: "Ac", covalent_radius: 2.15, mass: 227.0, vdw_radius: 0.0 },
    ElementData { number: 90, name: "thorium", symbol: "Th", covalent_radius: 2.06, mass: 232.0381, vdw_radius: 0.0 },
    ElementData { number: 91, name: "protactinium", symbol: "Pa", covalent_radius: 2.0, mass: 231.0359, vdw_radius: 0.0 },
    ElementData { number: 92, name: "uranium", symbol: "U", covalent_radius: 1.96, mass: 238.0289, vdw_radius: 0.0 },
];

/// Static element database, built at compile time from hardcoded data.
static SPECIES_DATA: Lazy<HashMap<&'static str, &'static ElementData>> =
    Lazy::new(|| ELEMENTS.iter().map(|e| (e.symbol, e)).collect());

/// Looks up an element property by symbol.
///
/// # Arguments
/// * `symbol` - Element symbol (e.g. "C", "Fe", "X" for ghost)
/// * `property` - One of: "number", "name", "symbol", "covalent_radius", "mass", "vdw_radius"
///
/// # Returns
/// The property value as `f64`.
///
/// # Panics
/// Panics if the element is not found or the property is invalid.
pub fn get_property(symbol: &str, property: &str) -> f64 {
    if is_ghost(symbol) {
        return ghost_row(property);
    }
    let data = SPECIES_DATA
        .get(symbol)
        .unwrap_or_else(|| panic!("No species with symbol {symbol:?}"));
    match property {
        "number" => data.number as f64,
        "covalent_radius" => data.covalent_radius,
        "mass" => data.mass,
        "vdw_radius" => data.vdw_radius,
        _ => panic!("Unknown property: {property:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_carbon_properties() {
        assert!((get_property("C", "covalent_radius") - 0.77).abs() < 1e-10);
        assert!((get_property("C", "mass") - 12.0107).abs() < 1e-4);
        assert_eq!(get_property("C", "number"), 6.0);
    }

    #[test]
    fn test_hydrogen_properties() {
        assert!((get_property("H", "covalent_radius") - 0.38).abs() < 1e-10);
        assert!((get_property("H", "mass") - 1.0079).abs() < 1e-4);
    }

    #[test]
    fn test_ghost_atom() {
        assert!(is_ghost("X"));
        assert!(is_ghost("BQ"));
        assert!(is_ghost("ghost"));
        assert!(!is_ghost("C"));
        assert!(!is_ghost("H"));
        assert_eq!(get_property("X", "mass"), 0.0);
        assert_eq!(get_property("X", "covalent_radius"), 0.0);
    }

    #[test]
    fn test_iron_properties() {
        assert!((get_property("Fe", "covalent_radius") - 1.25).abs() < 1e-10);
        assert!((get_property("Fe", "mass") - 55.845).abs() < 1e-3);
    }

    #[test]
    fn test_uranium_properties() {
        assert!((get_property("U", "covalent_radius") - 1.96).abs() < 1e-10);
        assert!((get_property("U", "mass") - 238.0289).abs() < 1e-3);
    }

    #[test]
    #[should_panic(expected = "No species with symbol")]
    fn test_unknown_element() {
        get_property("Xx", "mass");
    }
}
