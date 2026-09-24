# frozen_string_literal: true

# The Ruby tests under line coverage, for CI.
#
# What this holds is that a refusal, a branch or a value object does not
# quietly stop being exercised. Line coverage is not correctness, and the
# subprocess layer (`Runner::Child`, `Preflight`) does not appear here at
# all: those are covered by the tests that spawn them. What is here is the
# part of lib/torobi this process reaches, and a floor below where it sits.
#
#   ruby -Ilib -Itest tools/coverage.rb
require "coverage"

# Below this, something stopped being tested rather than being deleted.
# Overridable so the gate itself can be exercised without moving the floor.
FLOOR = Float(ENV.fetch("TOROBI_COVERAGE_FLOOR", "97.0"))

ROOT = File.expand_path("..", __dir__)

Coverage.start(lines: true)

# Registered before Minitest's own `at_exit`, so it runs after the suite
# has. It only ever lowers the status: a test that failed is not rescued
# by coverage being fine.
at_exit do
  rows = Coverage.result.filter_map do |path, data|
    next unless path.include?("/lib/torobi")

    lines = data[:lines].compact
    next if lines.empty?

    covered = lines.count(&:positive?)
    [100.0 * covered / lines.size, covered, lines.size, path.sub("#{ROOT}/", "")]
  end
  rows.sort_by { |row| [row[0], row[3]] }.each do |pct, covered, total, path|
    warn format("%6.2f%%  %4d/%-4d  %s", pct, covered, total, path)
  end
  covered = rows.sum { |row| row[1] }
  total = rows.sum { |row| row[2] }
  overall = total.zero? ? 0.0 : 100.0 * covered / total
  warn format("line coverage: %.2f%% (%d/%d, floor %.2f%%)", overall, covered, total, FLOOR)
  exit 1 if overall < FLOOR
end

$LOAD_PATH.unshift File.join(ROOT, "lib")
$LOAD_PATH.unshift File.join(ROOT, "test")
require "test_helper"
Dir[File.join(ROOT, "test/**/*_test.rb")].each { |file| require file }
