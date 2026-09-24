# frozen_string_literal: true

require_relative "test_helper"

# What the probe says when MLX cannot start. The probe runs in a subprocess
# and `errors_test` holds that it is asked once; its failure paths are the
# ones that never happen on a machine where MLX works, so they are held to
# directly.
class PreflightTest < Minitest::Test
  def test_a_probe_reason_is_the_childs_own_words
    output = "-e:12:in '<main>': MLX cannot start on this machine\n"

    assert_equal "MLX cannot start on this machine",
                 Torobi::Preflight::Probe.reason_from(output)
  end

  def test_a_probe_that_died_silently_says_what_that_means
    reason = Torobi::Preflight::Probe.reason_from("")

    assert_match(/ended a probe process rather than raising/, reason)
  end
end
