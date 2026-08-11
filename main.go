// Command lawnotation-iaa is the command-line front end for the IAA tool.
// Everything it does lives in the iaa package, which host applications can
// also import directly and mount with iaa.Handler().
package main

import "github.com/MaastrichtU-BISS/lawnotation-iaa/iaa"

func main() { iaa.RunCLI() }
