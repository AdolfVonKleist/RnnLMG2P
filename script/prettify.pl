#!/usr/bin/perl -w
use strict;
use warnings;
use utf8;
binmode STDIN, ':utf8';
binmode STDOUT, ':utf8';

my $rev = "false";
my $gpdelim = "}";
if (@ARGV > 0) { $rev = $ARGV[0]; }
if (@ARGV == 2) { $gpdelim = $ARGV[1]; }

while (<STDIN>) {
    chomp;
    next if m/^\s/;
    next if m/^Start/;

    @_ = split (/\s+/);
    my @g;
    my @p;
    my $weight = pop (@_);
    for my $tok (@_) {
	if ($tok eq "</s>") {
	    push (@g, "");
	    #push (@p, $tok);
	} else {
	    my ($l, $s) = split (/${gpdelim}/,$tok);
	    $l =~ s/\|//g;
	    $s =~ s/\|/ /g;
	    if ($rev eq "true") {
		$l = scalar reverse $l;
		$s = scalar (join (" ", reverse (split (/ /, $s))));
	    }
	    push (@g, $l);
	    if ($s ne "_") {
		push (@p, $s);
	    }
	}
    }
    print join ("", @g)."\t".join (" ", @p)."\t".$weight."\n";
}
