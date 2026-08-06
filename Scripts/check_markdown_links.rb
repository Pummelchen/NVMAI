#!/usr/bin/env ruby
# frozen_string_literal: true

# Validate that local relative links inside Markdown files resolve to real
# files. External (http/https), anchor (#...) and mailto links are skipped.
# Fails with a non-zero exit code when any local link is broken, so CI can
# gate on it.

require 'set'

Encoding.default_external = Encoding::UTF_8
Encoding.default_internal = Encoding::UTF_8

root = File.expand_path('..', __dir__)
failures = 0
checked = 0

Dir.glob(File.join(root, '**', '*.md')).sort.each do |file|
  next if file.include?('/.build/')
  next if file.include?('/.qwen/')

  dir = File.dirname(file)
  File.readlines(file).each_with_index do |line, index|
    line.scan(/\[[^\]]*\]\(([^)]+)\)/) do |match|
      target = match[0].strip
      next if target.empty?
      next if target.start_with?('http://', 'https://', '#', 'mailto:')

      target = target.split('#').first.to_s.gsub('%20', ' ')
      next if target.empty?

      path = File.expand_path(target, dir)
      checked += 1
      next if File.exist?(path)

      puts "#{file.delete_prefix(root + '/')}:#{index + 1}: broken link '#{target}'"
      failures += 1
    end
  end
end

puts "Checked #{checked} local link(s); #{failures} broken."
exit(failures.zero? ? 0 : 1)
