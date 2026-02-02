//! Integration tests for Legion.
//!
//! These tests verify the complete Legion system works together.

use super::*;

#[test]
fn test_legion_with_custom_config() {
    let legion = Legion::new(LegionConfig {
        agent_count: 8,
        default_quality: 0.9,
        progressive_refinement: true,
        consensus_strategy: ConsensusStrategy::WeightedMajority,
        timeout: std::time::Duration::from_secs(60),
    });

    assert!(legion.is_ok());
    let legion = legion.expect("Failed to create legion");
    assert_eq!(legion.agent_count(), 8);
}

#[test]
fn test_context_distribution() {
    let legion = Legion::new(LegionConfig::default()).expect("Failed to create legion");

    // Add context
    let ctx = legion.context();
    ctx.add_fragment("Important system prompt", 1.0);
    ctx.add_fragment("User context", 0.8);
    ctx.add_fragment("Historical data", 0.5);
    ctx.add_fragment("Low priority info", 0.2);

    // Strategic agents (25%) should get 1 fragment
    let strategic_frags = ctx.fragments_for_fraction(0.25);
    assert_eq!(strategic_frags.len(), 1);

    // Reflective agents (100%) should get all fragments
    let reflective_frags = ctx.fragments_for_fraction(1.0);
    assert_eq!(reflective_frags.len(), 4);
}

#[test]
fn test_frequency_band_assignment() {
    let legion = Legion::new(LegionConfig {
        agent_count: 5,
        ..Default::default()
    }).expect("Failed to create legion");

    // With 5 agents, we should have one at each frequency band
    let agents = legion.agents.read();

    // First agent (index 0) at 20% = Strategic (since 0.2 <= 0.25)
    assert_eq!(agents[0].frequency_band(), FrequencyBand::Strategic);

    // Last agent (index 4) at 100% = Reflective (since 1.0 > 0.90)
    assert_eq!(agents[4].frequency_band(), FrequencyBand::Reflective);
}
