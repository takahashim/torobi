# frozen_string_literal: true

source "https://rubygems.org"

gemspec

# rubocop asks for `parallel >= 1.10` and would take the newest, which is
# 2.x and wants Ruby 3.3. The gemspec claims 3.2 and CI holds it to that,
# so the lint's dependency is not allowed to raise the floor.
gem "parallel", "< 2"
