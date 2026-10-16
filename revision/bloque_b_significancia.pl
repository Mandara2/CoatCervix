use strict; use warnings; use List::Util qw(sum min max);
sub lgam { my $x=shift; my @c=(76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5);
  my $y=$x; my $t=$x+5.5; $t-=($x+0.5)*log($t); my $s=1.000000000190015; for my $c(@c){$y++; $s+=$c/$y} return -$t+log(2.5066282746310005*$s/$x); }
sub betacf { my($a,$b,$x)=@_; my($qab,$qap,$qam)=($a+$b,$a+1,$a-1); my $c=1; my $d=1-$qab*$x/$qap; $d=1e-30 if abs($d)<1e-30; $d=1/$d; my $h=$d;
  for my $m(1..300){ my $m2=2*$m; my $aa=$m*($b-$m)*$x/(($qam+$m2)*($a+$m2)); $d=1+$aa*$d; $d=1e-30 if abs($d)<1e-30; $c=1+$aa/$c; $c=1e-30 if abs($c)<1e-30; $d=1/$d; $h*=$d*$c;
    $aa=-($a+$m)*($qab+$m)*$x/(($a+$m2)*($qap+$m2)); $d=1+$aa*$d; $d=1e-30 if abs($d)<1e-30; $c=1+$aa/$c; $c=1e-30 if abs($c)<1e-30; $d=1/$d; my $del=$d*$c; $h*=$del; last if abs($del-1)<1e-12 } return $h; }
sub ibeta { my($a,$b,$x)=@_; return 0 if $x<=0; return 1 if $x>=1; my $bt=exp(lgam($a+$b)-lgam($a)-lgam($b)+$a*log($x)+$b*log(1-$x));
  return $x<($a+1)/($a+$b+2) ? $bt*betacf($a,$b,$x)/$a : 1-$bt*betacf($b,$a,1-$x)/$b; }
sub mean { sum(@_)/@_ } sub var { my $m=mean(@_); sum(map {($_-$m)**2} @_)/(@_-1) }
sub welch { my($x,$y)=@_; my($vx,$vy)=(var(@$x)/@$x, var(@$y)/@$y); my $t=(mean(@$x)-mean(@$y))/sqrt($vx+$vy);
  my $df=($vx+$vy)**2/($vx**2/(@$x-1)+$vy**2/(@$y-1)); return (ibeta($df/2,0.5,$df/($df+$t*$t)), $t, $df); }
sub ranks { my @v=@_; my @o=sort {$v[$a]<=>$v[$b]} 0..$#v; my @r; my $i=0; while($i<@o){ my $j=$i; $j++ while $j+1<@o && $v[$o[$j+1]]==$v[$o[$i]]; my $avg=($i+$j)/2+1; $r[$o[$_]]=$avg for $i..$j; $i=$j+1 } return @r; }
sub mw { my($x,$y)=@_; my @all=(@$x,@$y); my @r=ranks(@all); my $n=@$x; my $obs=sum(@r[0..$n-1]); my $exp=$n*(@all+1)/2; my $d0=abs($obs-$exp);
  my ($cnt,$tot)=(0,0); my @idx=0..$#all; my @comb; my $rec; $rec=sub{ my($s,@c)=@_; if(@c==$n){ $tot++; my $rs=sum(@r[@c]); $cnt++ if abs($rs-$exp)>=$d0-1e-9; return } for my $i($s..$#all){ $rec->($i+1,@c,$i) } }; $rec->(0); return $cnt/$tot; }
sub holm { my @p=@_; my @o=sort {$p[$a]<=>$p[$b]} 0..$#p; my @adj; my $prev=0; for my $k(0..$#o){ $prev=max($prev,min(1,$p[$o[$k]]*(@p-$k))); $adj[$o[$k]]=$prev } return @adj; }
my %coat=(acc=>[0.8133,0.7956,0.8089,0.7644,0.7689], f1=>[0.7810,0.7581,0.7728,0.7205,0.7232]);
my @mods=('ConvNeXt-Large','Swin-Base','ResNet50','VGG16','DenseNet201','VGG19');
my %b=( 'ConvNeXt-Large'=>{acc=>[.7733,.7600,.7511,.7778,.7333],f1=>[.7284,.7100,.6933,.7298,.6539]},
 'Swin-Base'=>{acc=>[.7333,.8000,.7422,.7511,.7644],f1=>[.6694,.7619,.6817,.7257,.7193]},
 'VGG16'=>{acc=>[.7289,.8133,.6933,.7200,.7244],f1=>[.6744,.7833,.6122,.6482,.6457]},
 'VGG19'=>{acc=>[.7333,.7511,.7067,.7156,.7067],f1=>[.6721,.6883,.6368,.6322,.6445]},
 'DenseNet201'=>{acc=>[.7600,.7156,.7289,.7200,.7556],f1=>[.7185,.6858,.6643,.6719,.7159]},
 'ResNet50'=>{acc=>[.7689,.7511,.7467,.7111,.7556],f1=>[.7262,.7123,.6967,.6513,.7106]} );
for my $met('acc','f1'){ my(@pw,@pm,@dif,@t,@df); for my $m(@mods){ my($p,$t,$df)=welch($coat{$met},$b{$m}{$met}); push @pw,$p; push @t,$t; push @df,$df; push @pm, mw($coat{$met},$b{$m}{$met}); push @dif, mean(@{$coat{$met}})-mean(@{$b{$m}{$met}}) }
  my @hw=holm(@pw); my @hm=holm(@pm); print "\n== ",uc($met)," | modelo | dif | t | df | p Welch | p MW | p Welch Holm | p MW Holm\n";
  printf "%s | %+.4f | %.2f | %.1f | %.4f | %.4f | %.4f | %.4f\n", $mods[$_],$dif[$_],$t[$_],$df[$_],$pw[$_],$pm[$_],$hw[$_],$hm[$_] for 0..$#mods; }
