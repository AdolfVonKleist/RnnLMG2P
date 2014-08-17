#!/usr/bin/perl -w
use strict;
use warnings;


while (<STDIN>) {
    chomp;
    @_ = split (/\s+/);
    my @g;
    my @p;
    my $weight = pop (@_);
    for my $tok (@_) {
	if ($tok eq "</s>") {
	    push (@g, "");
	    push (@p, $tok);
	} else {
	    my ($l, $s) = split (/\}/,$tok);
	    $l =~ s/\|//g;
	    $s =~ s/\|/ /g;
	    push (@g, $l);
	    if ($s ne "_") {
		push (@p, $s);
	    }
	}
    }
    print join ("", @g)."\t".join (" ", @p)."\t".$weight."\n";
}
